import os
import sys
import cv2
import json
import time
import copy
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from datetime import datetime
import torchvision.transforms as transforms
from ultralytics import YOLO 
from glob import glob

# 请确保这些路径与您的工程匹配
from HccePose.bop_loader import BopDataset
from HccePose.network_model import HccePose_PnPNet_Net, load_checkpoint
from HccePose.tools.rot_reps import rot6d_to_mat_batch
from kasal.bop_toolkit_lib.inout import load_ply

# ==========================================================
# 1. 预处理与可视化工具函数
# ==========================================================
def pad_square_fp32(GT_Bbox, padding_ratio=1.5):
    center_x = GT_Bbox[0] + 0.5 * GT_Bbox[2]
    center_y = GT_Bbox[1] + 0.5 * GT_Bbox[3]
    w = max(GT_Bbox[2], GT_Bbox[3]) * padding_ratio
    return np.array([center_x - w/2, center_y - w/2, w, w])

def crop_square_resize(img, Bbox, crop_size=256, interpolation=cv2.INTER_LINEAR):
    center_x = Bbox[0] + 0.5 * Bbox[2]
    center_y = Bbox[1] + 0.5 * Bbox[3]
    w_2 = Bbox[2] / 2
    pts1 = np.float32([[center_x - w_2, center_y - w_2], 
                       [center_x - w_2, center_y + w_2], 
                       [center_x + w_2, center_y - w_2]])
    pts2 = np.float32([[0, 0], [0, crop_size], [crop_size, 0]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (crop_size, crop_size), flags=interpolation)

composed_transforms_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def render_mesh_crop(mesh, R, t, K, img_w, img_h, bbox, crop_size=256):
    """根据预测位姿渲染全图,并按照bbox裁剪成256x256"""
    mesh_copy = copy.deepcopy(mesh) if 'copy' in globals() else mesh.copy()
    mesh_copy.paint_uniform_color([0.5, 0.5, 0.5]) 
    mesh_copy.compute_vertex_normals()
    mesh_copy.rotate(R, center=(0, 0, 0))
    mesh_copy.translate(t)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=img_w, height=img_h)
    vis.add_geometry(mesh_copy)
    vis.poll_events()
    vis.update_renderer()
    
    ctr = vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, K[0,0], K[1,1], K[0,2], K[1,2])
    param.extrinsic = np.eye(4) 
    success = ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    if not success:
        # 如果设置失败，通常是因为窗口大小被系统强制改变了
        print(f"Warning: Camera calibration failed! Successful: {success}")

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) 
    opt.light_on = True

    vis.poll_events()
    vis.update_renderer()
    img_buffer = vis.capture_screen_float_buffer(do_render=True)
    rendered_img = (np.asarray(img_buffer) * 255).astype(np.uint8)
    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
    vis.destroy_window()
    
    if rendered_img.shape[0] != img_h or rendered_img.shape[1] != img_w:
        rendered_img = cv2.resize(rendered_img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    return crop_square_resize(rendered_img, bbox, crop_size, cv2.INTER_LINEAR)

# ==========================================================
# 2. 您的多目标并行网络 (增加可视化特征图输出)
# ==========================================================
class MultiObjectPoseNet(torch.nn.Module):
    def __init__(self, hcce_list_dict):
        super().__init__()
        self.models = torch.nn.ModuleDict({str(k): v for k, v in hcce_list_dict.items()})
        self.streams = {str(k): torch.cuda.Stream() for k in hcce_list_dict.keys()}
        
    def forward(self, img_batch, cam_K, bbox_batch, img_size_tensor, class_ids):
        device = img_batch.device
        N = img_batch.shape[0]
        
        pred_rot_6d_all = torch.zeros((N, 6), device=device, dtype=torch.float32)
        pred_trans_all = torch.zeros((N, 3), device=device, dtype=torch.float32)
        
        # 增加：为可视化保存 Mask 和 HCCE
        pred_mask_all = None
        pred_front_code_all = None
        pred_back_code_all = None
        
        unique_classes = torch.unique(class_ids)
        
        for cls in unique_classes:
            str_cls = str(cls.item())
            if str_cls not in self.models:
                continue
            mask = (class_ids == cls)
            
            with torch.cuda.stream(self.streams[str_cls]):
                sub_results = self.models[str_cls].inference_batch(
                    img_batch[mask], cam_K[mask], bbox_batch[mask], img_size_tensor[mask]
                )
                
                # 动态初始化可视化 Tensor
                if pred_mask_all is None:
                    _, h, w = sub_results['pred_mask'].shape
                    pred_mask_all = torch.zeros((N, h, w), device=device, dtype=torch.float32)
                    _, c_f, h_f, w_f = sub_results['pred_front_code'].shape
                    pred_front_code_all = torch.zeros((N, c_f, h_f, w_f), device=device, dtype=torch.float32)
                    pred_back_code_all = torch.zeros((N, c_f, h_f, w_f), device=device, dtype=torch.float32)
                    
                pred_rot_6d_all[mask] = sub_results['pred_rot_6d'].to(torch.float32)
                pred_trans_all[mask] = sub_results['pred_trans'].to(torch.float32)
                pred_mask_all[mask] = sub_results['pred_mask'].to(torch.float32)
                pred_front_code_all[mask] = sub_results['pred_front_code'].to(torch.float32)
                pred_back_code_all[mask] = sub_results['pred_back_code'].to(torch.float32)
                
        torch.cuda.synchronize()
        return {
            'pred_rot_6d': pred_rot_6d_all, 
            'pred_trans': pred_trans_all,
            'pred_mask': pred_mask_all,
            'pred_front_code': pred_front_code_all,
            'pred_back_code': pred_back_code_all
        }


# ==========================================================
# 3. 核心推理主程序
# ==========================================================
"""
PYTHONPATH=. xvfb-run -a python /HCCEPose/scripts/s5_inference_pipeline.py
"""

if __name__ == '__main__':
    # ------------------ 路径与参数配置 ------------------
    CUDA_DEVICE = '0'
    device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
    now_stamp = datetime.now()
    now_str = now_stamp.strftime('%Y-%m-%d_%H:%M:%S')
    
    dataset_path = '/HCCEPose/datasets/dataset_name'
    net_name = 'convnext'
    obj_id_list = [1, 2, 3, 4, 5]
    padding_ratio = 1.5
    
    YOLO_MODEL_PATH = "/HCCEPose/output/dataset_name/detection/obj_s/2026-04-23_14:48:50/train/weights/best.pt"
    IMAGE_PATH = "/HCCEPose/pipeline/images/*.bmp"
    OUTPUT_DIR = f"/HCCEPose/pipeline/inference_results/{now_str}"
    
    img_paths = glob(IMAGE_PATH)
    
    # 手动指定的权重路径 (或者按您的规则自动搜寻)
    checkpoint_map = {
        1: '/HCCEPose/output/dataset_name/pose_estimation/2026-05-06_17:28:54/obj_01/best_score/',
        2: '/HCCEPose/output/dataset_name/pose_estimation/2026-05-07_09:30:13/obj_02/best_score/',
        3: '/HCCEPose/output/dataset_name/pose_estimation/2026-05-07_09:30:13/obj_03/best_score/',
        4: '/HCCEPose/output/dataset_name/pose_estimation/2026-05-07_09:30:13/obj_04/best_score/',
        5: '/HCCEPose/output/dataset_name/pose_estimation/2026-05-07_09:30:13/obj_05/best_score/',
    }

    # ------------------ 1. 加载 TRPN-HCCE 模型与 Mesh ------------------
    print("Loading models and meshes...")
    bop_dataset_item = BopDataset(dataset_path)
    hcce_list_dict = {}
    obj_meshes = {}
    
    for obj_id in obj_id_list:
        idx = bop_dataset_item.obj_id_list.index(obj_id)
        obj_path = bop_dataset_item.obj_model_list[idx]
        obj_info = bop_dataset_item.obj_info_list[idx]
        
        if obj_id in checkpoint_map:
            best_save_path = checkpoint_map[obj_id]
        else:
            best_save_path = os.path.join(dataset_path, 'HccePose', f'obj_{obj_id:02d}', 'best_score')
        
        if not os.path.exists(best_save_path):
            print(f"Skipping Obj {obj_id}: Checkpoint not found at {best_save_path}")
            continue
            
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']], dtype=np.float32)).to(device)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']], dtype=np.float32)).to(device)
        
        net = HccePose_PnPNet_Net(net=net_name, input_channels=3, min_xyz=min_xyz, size_xyz=size_xyz)
        load_checkpoint(best_save_path, net, CUDA_DEVICE=CUDA_DEVICE)
        net = net.to(device).eval()
        hcce_list_dict[str(obj_id)] = net
        
        # 加载用于渲染的 Mesh
        obj_meshes[str(obj_id)] = o3d.io.read_triangle_mesh(obj_path)
        print(f"Loaded Obj {obj_id} successfully.")
        
    multi_model = MultiObjectPoseNet(hcce_list_dict).to(device)
    multi_model.eval()

    # ------------------ 2. 读取图像与相机内参 ------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (768, 512), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_bgr.shape[:2]
        
        # 固定相机内参
        cam_K_np = np.array([867.8339, 0.0, 382.9796, 0.0, 868.0221, 259.3721, 0.0, 0.0, 1.0]).reshape(3, 3).astype(np.float32)
        
        # ------------------ 3. YOLO 目标检测与保存 JSON ------------------
        print("Running YOLO detection...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        results = yolo_model.predict(img_rgb, conf=0.6, save=True, project=os.path.join(OUTPUT_DIR, img_name), name='predict_results')[0] # 置信度阈值可调
        
        detections_log = []
        tensor_list_rgb = []
        tensor_list_bbox = []
        class_id_list = []
        crop_bgr_list = []
        
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item()) + 1
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            raw_bbox = [float(x1), float(y1), float(w), float(h)]
            # 只保留我们加载了模型的类别
            if str(cls_id) not in hcce_list_dict:
                continue
            
            detections_log.append({
                "obj_idx": idx,
                "class_id": cls_id,
                "confidence": conf,
                "bbox_xywh": raw_bbox
            })
            
            # 裁剪并转换为 Tensor
            padded_bbox = pad_square_fp32(np.array(raw_bbox), padding_ratio=padding_ratio)
            roi_img = crop_square_resize(img_rgb, padded_bbox, crop_size=256)
            roi_tensor = composed_transforms_img(Image.fromarray(roi_img))
            
            tensor_list_rgb.append(roi_tensor)
            tensor_list_bbox.append(padded_bbox)
            class_id_list.append(cls_id)
            crop_bgr_list.append(cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
            
        # 保存检测到的 bbox 到 JSON
        json_path = os.path.join(OUTPUT_DIR, img_name, "detections.json")
        with open(json_path, 'w') as f:
            json.dump(detections_log, f, indent=4)
        print(f"Saved YOLO detections to {json_path}")
        
        if len(tensor_list_rgb) == 0:
            print("No valid objects detected. Exiting.")
            sys.exit()

        # ------------------ 4. 组装全局 Batch 送入 MultiObjectPoseNet ------------------
        print(f"Running TRPN-HCCE pose estimation for {len(tensor_list_rgb)} objects...")
        batch_rgb = torch.stack(tensor_list_rgb).to(device)
        batch_bbox = torch.tensor(np.array(tensor_list_bbox), dtype=torch.float32).to(device)
        batch_class_ids = torch.tensor(class_id_list, dtype=torch.long).to(device)
        
        batch_K = torch.from_numpy(cam_K_np).unsqueeze(0).repeat(len(tensor_list_rgb), 1, 1).to(device)
        batch_sz = torch.tensor([img_h, img_w], dtype=torch.float32).unsqueeze(0).repeat(len(tensor_list_rgb), 1).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # 一键调用您封装的多模型并行网络
                pred_res = multi_model(batch_rgb, batch_K, batch_bbox, batch_sz, batch_class_ids)
                
        # 解析输出结果
        pred_rot_mats = rot6d_to_mat_batch(pred_res['pred_rot_6d']).cpu().numpy()
        pred_trans = pred_res['pred_trans'].cpu().numpy()
        pred_masks = pred_res['pred_mask'].cpu().numpy()

        # ------------------ 5. 解码密集图与生成最终可视化 ------------------
        for i, det in enumerate(detections_log):
            img_point_cloud_vis = img_bgr.copy()
            obj_idx = det["obj_idx"]
            cls_id = str(det["class_id"])
            bbox = tensor_list_bbox[i]
            
            # 1. 使用对应类别的网络对 HCCE 隐空间编码进行解码（恢复 0-255 的坐标图）
            net = hcce_list_dict[cls_id]
            front_code = pred_res['pred_front_code'][i:i+1] # 保持 batch 维度
            back_code = pred_res['pred_back_code'][i:i+1]
            with torch.no_grad():
                front_decoded = net.hcce_decode(front_code.permute(0,2,3,1)).squeeze(0).cpu().numpy() * 255
                back_decoded = net.hcce_decode(back_code.permute(0,2,3,1)).squeeze(0).cpu().numpy() * 255
                
            # 2. 生成各种切片图像
            crop_rgb = crop_bgr_list[i]
            mask_img = (pred_masks[i] > 0.5).astype(np.uint8) * 255
            mask_img_3c = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            mask_img_3c = cv2.resize(mask_img_3c, (256, 256), interpolation=cv2.INTER_NEAREST)
            front_img = cv2.cvtColor(front_decoded.astype(np.uint8), cv2.COLOR_RGB2BGR)
            front_img = cv2.resize(front_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            back_img = cv2.cvtColor(back_decoded.astype(np.uint8), cv2.COLOR_RGB2BGR)
            back_img = cv2.resize(back_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            # 3. 提取位姿并渲染 3D 模型
            R = pred_rot_mats[i].reshape(3, 3)
            T = pred_trans[i].reshape(3, 1)
            mesh = obj_meshes[cls_id]
            rendered_crop = render_mesh_crop(mesh, R, T, cam_K_np, img_w, img_h, bbox)
            
            # 4. 横向拼接 5 拼图
            composite_img = np.hstack([crop_rgb, front_img, back_img, mask_img_3c, rendered_crop])
            bottom_margin = 40
            composite_img = cv2.copyMakeBorder(
                composite_img, 0, bottom_margin, 0, 0, 
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            cv2.putText(composite_img, f"Obj:{obj_idx} Cls:{cls_id}", (10, 256 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            comp_save_name = f"{obj_idx}_{cls_id}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name, comp_save_name), composite_img)
            print(f"Saved: {comp_save_name}")
            
            # 5. 在原图上画绿色点云投影
            # MAX_POINTS = 1500
            pts_3d = np.asarray(mesh.vertices)
            # if pts_3d.shape[0] > MAX_POINTS:
            #     indices = np.linspace(0, pts_3d.shape[0] - 1, MAX_POINTS, dtype=int)
            #     pts_3d = pts_3d[indices]
            pts_3d_transformed = (R @ pts_3d.T + T).T
            pts_2d, _ = cv2.projectPoints(pts_3d_transformed, np.zeros((3,1)), np.zeros((3,1)), cam_K_np, None)
            pts_2d = pts_2d.squeeze().astype(int)
            for pt in pts_2d:
                if 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
                    img_point_cloud_vis[pt[1], pt[0]] = (0, 255, 0)
            
            # 1. 获取模型原始的 3D 边界框的 8 个角点
            min_b = mesh.get_min_bound()
            max_b = mesh.get_max_bound()
            corners_3d = np.array([
                [min_b[0], min_b[1], min_b[2]], [max_b[0], min_b[1], min_b[2]],
                [max_b[0], max_b[1], min_b[2]], [min_b[0], max_b[1], min_b[2]],
                [min_b[0], min_b[1], max_b[2]], [max_b[0], min_b[1], max_b[2]],
                [max_b[0], max_b[1], max_b[2]], [min_b[0], max_b[1], max_b[2]],
            ])
            
            # 2. 将 8 个角点进行刚体变换并投影到 2D
            corners_3d_transformed = (R @ corners_3d.T + T).T
            pts_2d, _ = cv2.projectPoints(corners_3d_transformed, np.zeros((3,1)), np.zeros((3,1)), cam_K_np, None)
            pts_2d = pts_2d.squeeze().astype(int)
            
            # 3. 绘制 12 条连接线构成一个 3D 立体框
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            for e_i, e_j in edges:
                pt1, pt2 = tuple(pts_2d[e_i]), tuple(pts_2d[e_j])
                # 边界安全检查
                if 0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h and 0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h:
                    cv2.line(img_point_cloud_vis, pt1, pt2, (255, 255, 0), 1)

            # 6. 保存最终的点云大图
            pc_save_name = f"{obj_idx}_pointcloud_projection.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name, pc_save_name), img_point_cloud_vis)
            print(f"Saved full image point cloud projection to {pc_save_name}")
    print("Inference Pipeline Completed!")