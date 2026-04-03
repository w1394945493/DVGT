import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from typing import Dict
from iopath.common.file_io import g_pathmgr
from safetensors.torch import load_file
import logging

from dvgt.models.backbones.dvgt1_aggregator import DVGT1Aggregator
from dvgt.models.heads.dvgt1_ego_pose_head import DVGT1EgoPoseHead
from dvgt.models.heads.dpt_head import DPTHead
from dvgt.utils.pose_encoding import decode_pose
from dvgt.evaluation.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n

class DVGT1(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        patch_size=16, 
        embed_dim=1024,
        depth=24, 
        patch_embed="dinov3_vitl16",
        dino_v3_weight_path="ckpt/dino_v3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        enable_ego_pose=True, 
        enable_point=True, 
        frames_chunk_size=None, # DPT head: Number of frames to process in each chunk.
    ):
        super().__init__()

        self.frames_chunk_size = frames_chunk_size
        self.enable_ego_pose = enable_ego_pose

        dim_in = 3 * embed_dim

        self.aggregator = DVGT1Aggregator(
            patch_size=patch_size, embed_dim=embed_dim, depth=depth, 
            patch_embed=patch_embed, dino_v3_weight_path=dino_v3_weight_path,
        )

        self.ego_pose_head = DVGT1EgoPoseHead(dim_in=dim_in) if enable_ego_pose else None
        self.point_head = DPTHead(dim_in=dim_in, patch_size=patch_size, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the DVGT model.

        Args:
            images (torch.Tensor): Input images with shape [T, V, 3, H, W] or [B, T, V, 3, H, W], in range [0, 1].
                B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Ego pose encoding with shape [B, T, V, 7] (from the last iteration)
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, T, V, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, T, V, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 5:
            images = images.unsqueeze(0)
            
        aggregated_tokens_list, patch_start_idx = self.aggregator(images) # (1 16 8 3 288 512) T=16

        predictions = {}

        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if self.ego_pose_head is not None:                                # 4: (1 16 7) T=16 # 自车位姿
                ego_pose_enc_list = self.ego_pose_head(aggregated_tokens_list)
                predictions["absolute_ego_pose_enc"] = ego_pose_enc_list[-1]  # (1 16 7) # pose encoding of the last iteration
                predictions["absolute_ego_pose_enc_list"] = ego_pose_enc_list

            if self.point_head is not None:    # 全局3D点地图
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, frames_chunk_size=self.frames_chunk_size
                )
                predictions["points"] = pts3d                                 # (1 16 8 288 512 3)
                predictions["points_conf"] = pts3d_conf                       # (1 4 8 288 512)

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def custom_load_state_dict(self, checkpoint_conf: Dict):
        logging.info(f"Loading checkpoint from {checkpoint_conf}")
        
        if checkpoint_conf.get('direct_load_pretrained_weights_path', None):
            # use for fine-tuning
            with g_pathmgr.open(checkpoint_conf['direct_load_pretrained_weights_path'], "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            
            # Load model state
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            missing, unexpected = self.load_state_dict(
                model_state_dict, strict=checkpoint_conf.get('strict', True)
            )
            return missing, unexpected, None

        elif checkpoint_conf.get('load_pretrained_weights_path', None):
            state_dict = load_file(checkpoint_conf['load_pretrained_weights_path'], device="cpu")
            new_state_dict = {}
            for name, params in state_dict.items():
                # Rule 1: Retain the register_token.
                if 'aggregator.register_token' in name:
                    new_state_dict[name] = params

                # Rule 2: Retain the camera_token and copy its weights to the ego_pose_token based on the configuration.
                elif 'aggregator.camera_token' in name:
                    if self.enable_ego_pose:
                        new_name = name.replace('camera_token', 'ego_pose_token')
                        new_state_dict[new_name] = params

                # Rule 3: Rename frame_blocks to intra_view_blocks.
                elif 'aggregator.frame_blocks' in name:
                    new_name = name.replace('frame_blocks', 'intra_view_blocks')
                    new_state_dict[new_name] = params

                # Rule 4: Copy the parameters of the global_blocks to other block types according to the aa_order configuration.
                elif 'aggregator.global_blocks' in name:
                    new_name = name.replace('global_blocks', 'cross_view_blocks')
                    new_state_dict[new_name] = params
                    new_name = name.replace('global_blocks', 'cross_frame_blocks')
                    new_state_dict[new_name] = params

                # We do not load the decoder, force model to relearn from scratch.

            missing, unexpected = self.load_state_dict(
                new_state_dict, strict=checkpoint_conf.get('strict', True)
            )
            return missing, unexpected, None

        elif checkpoint_conf.get('resume_checkpoint_path', None):
            ckpt_path = checkpoint_conf['resume_checkpoint_path']

            with g_pathmgr.open(ckpt_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            
            # Load model state
            model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing, unexpected = self.load_state_dict(
                model_state_dict, strict=True
            )
            return missing, unexpected, checkpoint
        
        else:
            raise NotImplementedError("No checkpoint configuration provided.")

    def post_process_for_eval(
        self,
        predictions: Dict, 
        batch: Dict, 
        **kwargs,
    ) -> Dict:
        outputs = {}

        # eval point
        outputs['points_in_ego_first'] = predictions['points']
        outputs['points_in_ego_first_conf'] = predictions['points_conf']
        outputs['ray_depth'] = convert_point_in_ego_0_to_ray_depth_in_ego_n(predictions['points'], batch["ego_n_to_ego_first"])

        # eval pose
        outputs['ego_n_to_ego_first'], _ = decode_pose(predictions['absolute_ego_pose_enc'], pose_encoding_type="absT_quaR")

        # gt_scale_factor
        if kwargs.get("gt_scale_factor", 1.0) != 1.0:
            s = kwargs['gt_scale_factor']
            keys_to_scale_content = {
                'points_in_ego_first',
                'ray_depth',
            }
            keys_to_scale_translation = {
                'ego_n_to_ego_first',
            }
            for key in outputs.keys():
                if key in keys_to_scale_content:
                    outputs[key] /= s
                    batch[key] /= s
                elif key in keys_to_scale_translation:
                    outputs[key][..., :3, 3] /= s
                    batch[key][..., :3, 3] /= s
                    
        return outputs