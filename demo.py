import argparse

import torch
import os

from iopath.common.file_io import g_pathmgr

import numpy as np
from einops import rearrange

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dvgt.models.architectures.dvgt1 import DVGT1
# from dvgt.models.architectures.dvgt2 import DVGT2
from dvgt.utils.load_fn import load_and_preprocess_images
from dvgt.utils.pose_encoding import decode_pose
from dvgt.evaluation.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.visualization.utils import apply_sky_segmentation, depth_edge, points_to_normals, \
    normals_edge, visualize_ego_poses, process_and_filter_points, center_data



#=========================================#
# 将模型预测的3d点云 -> 过滤 -> 转成干净的点云用于可视化
def visualize_pred(predictions, args):
    B, T, V, H, W, _ = predictions['points'].shape
    assert B == 1, "Only batch size = 1 is supported for the visualization."
    #========================================================================#
    # 位姿解码
    pred_ego_n_to_ego_0, _ = decode_pose(predictions['absolute_ego_pose_enc'])

    #======================================#
    # 将点云转换成深度
    pred_points = predictions['points']
    pred_ray_depth_in_ego_n = convert_point_in_ego_0_to_ray_depth_in_ego_n(pred_points, pred_ego_n_to_ego_0)

    # Squeeze the batch dimension for visualization.
    pred_points = pred_points[0].cpu().numpy()
    pred_points_conf = predictions['points_conf'][0].cpu().numpy()
    pred_depth = pred_ray_depth_in_ego_n[0].cpu().numpy()       # for depth edge mask
    pred_ego_poses = pred_ego_n_to_ego_0[0].cpu().numpy()
    images = rearrange(predictions['images'][0].cpu().numpy(), 't v c h w -> t v h w c') * 255   # T, V, H, W, 3
    images = images.astype(np.uint8)

    # construct the combined mask
    combined_mask = np.ones((T, V, H, W), dtype=bool)   

    #======================================#
    # 置信度过滤
    if args.conf_threshold > 0:
        cutoff_value = np.percentile(pred_points_conf, args.conf_threshold)
        conf_mask = pred_points_conf >= cutoff_value
        combined_mask &= conf_mask
    #======================================#
    # 去除天空点
    if args.mask_sky:
        sky_mask = apply_sky_segmentation(pred_points_conf, images) 
        combined_mask &= sky_mask
    #======================================#
    # 边缘过滤：去掉深度突变(不连续的点)
    if args.use_edge_masks:
        # Applying edge masks
        edge_mask = np.ones_like(combined_mask)

        for t_idx in range(T):
            for v_idx in range(V):
                frame_pts = pred_points[t_idx, v_idx]  # (H, W, 3)
                frame_depth = pred_depth[t_idx, v_idx] # (H, W)
                frame_base_mask = combined_mask[t_idx, v_idx] # (H, W)

                # 1. Depth Edge Mask
                depth_edges = depth_edge(
                    frame_depth, 
                    rtol=args.edge_depth_rtol, 
                    mask=frame_base_mask
                )
                
                # 2. Normal Edge Mask
                normals, normals_mask = points_to_normals(
                    frame_pts, mask=frame_base_mask
                )
                normal_edges = normals_edge(
                    normals, tol=args.edge_normal_tol, mask=normals_mask
                )

                edge_mask[t_idx, v_idx] = ~(depth_edges | normal_edges)
            
            combined_mask &= edge_mask

    points_final, colors_final = process_and_filter_points(
        pred_points, images, combined_mask, args.max_depth, args.downsample_ratio
    )
    
    points_centered, poses_centered, center = center_data(points_final, pred_ego_poses)

    return [(points_centered, colors_final, "pred_point_cloud")], poses_centered


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Driving Scene Point Cloud Visualizer")
    
    parser.add_argument(
        "--model_name", type=str, default="DVGT1", choices=['DVGT1', 'DVGT2'])
    parser.add_argument(
        "--image_folder", type=str, default="visual_demo_examples/openscene_log-0104-scene-0007", help="Path to folder containing images"
    )    
    parser.add_argument("--start_frame", type=int, default=0, help="The start frame in the example autonomous video.")
    parser.add_argument("--end_frame", type=int, default=4, help="The end frame in the example autonomous video.")

    parser.add_argument('--downsample_ratio', type=float, default=-1, help="Random downsample ratio (0.0 to 1.0). Default: -1 (no downsampling).")
    parser.add_argument('--max_depth', type=float, default=-1, help="Maximum depth of points to visualize in meters. Default: -1 (no truncation).")

    parser.add_argument('--no_ego', action='store_true', help="Disable ego pose visualization.")
    parser.add_argument('--use_edge_masks', action='store_true', help="Enable depth and normal edge masks.")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")

    parser.add_argument('--edge_depth_rtol', type=float, default=0.1, help="Relative tolerance (rtol) for depth edge detection.")
    parser.add_argument('--edge_normal_tol', type=float, default=50, help="Angle tolerance (degrees) for normal edge detection.")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")

    
    return parser.parse_args()
    


if __name__=='__main__':

    args = parse_args()


    checkpoint_path = '/c20250502/wangyushen/Weights/dvgt/dvgt1.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    model = DVGT1()
    # model = DVGT2() # Let'2 try DVGT2
    with g_pathmgr.open(args.checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    #====================================================================================#
    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(args.image_folder, start_frame=args.start_frame, end_frame=args.end_frame).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            # Predict attributes including ego pose and point maps.
            predictions = model(images)
            # points = predictions['points'] # (1 16 8 288 512 3)
            point_clouds_to_show, poses = visualize_pred(predictions, args)

