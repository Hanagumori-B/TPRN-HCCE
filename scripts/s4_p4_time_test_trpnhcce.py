import os
import sys
import torch
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import ConcatDataset, DataLoader
from datetime import datetime

# 导入你原有的依赖，根据实际路径调整
from HccePose.bop_loader import BopDataset, TestBopDatasetBF_PnPNet
from HccePose.network_model import HccePose_PnPNet_Net, load_checkpoint
from kasal.bop_toolkit_lib.inout import load_ply

# ================= 数据集包装器 =================
class ObjIDWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, obj_id, use_gt_bbox=False, ratio=1.0):
        self.dataset = dataset
        self.obj_id = obj_id
        self.use_gt_bbox = use_gt_bbox 
        
        self.total_len = len(self.dataset)
        if ratio < 1.0:
            self.actual_len = max(1, int(self.total_len * ratio))
            # np.linspace 保证了均匀地从头到尾抽取指定数量的索引
            self.indices = np.linspace(0, self.total_len - 1, self.actual_len, dtype=int).tolist()
        else:
            self.actual_len = self.total_len
            self.indices = list(range(self.total_len))

    def __len__(self):
        return self.actual_len

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.dataset[real_idx]
        if self.use_gt_bbox:
            scene_id, image_id = self.get_meta(idx)
            score = 1.0  
            return (*data, scene_id, image_id, score, self.obj_id)
        else:
            return (*data, self.obj_id)
    
    def get_meta(self, idx):
        inner_ds = self.dataset 
        obj_key = 'obj_%s' % str(inner_ds.current_obj_id).rjust(6, '0')
        info_ = inner_ds.dataset_info['obj_info'][obj_key][idx]
        return int(info_['scene']), int(info_['image'])


if __name__ == '__main__':
    np.random.seed(0)
    
    # --- 基础配置 ---
    net_name = 'convnext'
    dataset_name = 'dataset_name'
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, '..', 'datasets', dataset_name)
    
    use_gt_bbox = True
    bbox_2D = None if use_gt_bbox else '/HCCEPose/datasets/dataset_name/test/gt_bbox2d.json'
    
    # 结果保存路径
    save_base_path = f'/HCCEPose/output/{dataset_name}/timing_stats'
    now_stamp = datetime.now()
    save_dir = os.path.join(save_base_path, net_name, now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_folder_name = 'test'
    
    obj_id_list = [1, 2, 3, 4, 5]
    
    checkpoint_map = {
        1: '/HCCEPose/output/dataset_name/pose_estimation3/2026-05-06_17:28:54/obj_01/best_score/',
        2: '/HCCEPose/output/dataset_name/pose_estimation3/2026-05-07_09:30:13/obj_02/best_score/',
        3: '/HCCEPose/output/dataset_name/pose_estimation3/2026-05-07_09:30:13/obj_03/best_score/',
        4: '/HCCEPose/output/dataset_name/pose_estimation3/2026-05-07_09:30:13/obj_04/best_score/',
        5: '/HCCEPose/output/dataset_name/pose_estimation3/2026-05-07_09:30:13/obj_05/best_score/',
    }
    CUDA_DEVICE = '0'
    padding_ratio = 1.5
    num_workers = 4
    EVAL_RATIO = 0.15 
    
    # --- 加载数据集和模型 ---
    bop_dataset_item = BopDataset(dataset_path)
    
    hcce_list_dict = {}
    wrapped_datasets = []
    
    print("\n>>> 正在初始化模型与加载预训练权重...")
    for obj_id in obj_id_list:
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        best_save_path = checkpoint_map[obj_id]
        
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        net = HccePose_PnPNet_Net(
            net=net_name, input_channels=3, 
            min_xyz=min_xyz, size_xyz=size_xyz
        )
        # 加载权重
        load_checkpoint(best_save_path, net, CUDA_DEVICE=CUDA_DEVICE)
        net = net.to('cuda:'+CUDA_DEVICE)
        net.eval()
        hcce_list_dict[str(obj_id)] = net
        
        # 构建数据集
        kwargs = {'padding_ratio': padding_ratio}
        if bbox_2D is not None: kwargs['bbox_2D'] = bbox_2D
        ds_item = TestBopDatasetBF_PnPNet(bop_dataset_item, dataset_folder_name, **kwargs)
        ds_item.update_obj_id(obj_id, obj_path)
        wrapped_datasets.append(ObjIDWrapperDataset(ds_item, obj_id, use_gt_bbox, EVAL_RATIO))

    mixed_dataset = ConcatDataset(wrapped_datasets)
    
    # 【关键改动】：废弃 ImageBatchSampler，使用标准的 batch_size=1
    # 这样每次取出的绝对只有一个实例，保证了测试环境的纯粹性
    mixed_loader = DataLoader(
        mixed_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n>>> 开始单实例精准耗时评测 (共 {len(mixed_loader)} 个实例)...")
    
    # ================= 性能统计容器 =================
    total_time = 0.0
    total_instances = 0
    
    class_time_sum = defaultdict(float)
    class_instance_count = defaultdict(int)
    image_time_sum = defaultdict(float)
    
    # CUDA 预热 (Warm-up)，防止首次调用的初始化开销污染计时
    dummy_input = torch.randn(1, 3, 256, 256).to('cuda:'+CUDA_DEVICE)
    dummy_bbox = torch.tensor([[0,0,100,100]]).to('cuda:'+CUDA_DEVICE)
    dummy_sz = torch.tensor([[640, 480]]).to('cuda:'+CUDA_DEVICE)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(5):
            _ = hcce_list_dict[str(obj_id_list[0])].inference_batch(dummy_input, torch.eye(3).unsqueeze(0).to('cuda:'+CUDA_DEVICE), dummy_bbox, dummy_sz)
    torch.cuda.synchronize()

    # ================= 开始推理与计时 =================
    for batch_data in tqdm(mixed_loader, desc="Inference", colour='green'):
        # 解包 (bs=1)
        (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, image_size, 
         cam_R_m2c, cam_t_m2c, scene_id, image_id, score, class_ids) = batch_data
        
        # 提取 ID 标示
        cls_id_str = str(int(class_ids[0].item()))
        img_key = f"{int(scene_id[0].item())}_{int(image_id[0].item())}"
        
        # 转移至 GPU
        rgb_c = rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
        Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True)
        cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True)
        image_size = image_size.to('cuda:'+CUDA_DEVICE, non_blocking=True)
        
        net = hcce_list_dict[cls_id_str]

        # 【核心计时逻辑】
        torch.cuda.synchronize() # 确保数据转移完成
        t_start = time.perf_counter() # 使用高精度时钟
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # 仅执行一个物体的网络前向与解码
                _ = net.inference_batch(rgb_c, cam_K, Bbox, image_size)
                
        torch.cuda.synchronize() # 确保 GPU 计算全部完成
        t_end = time.perf_counter()
        
        elapsed = t_end - t_start
        
        # 累加时间
        total_time += elapsed
        total_instances += 1
        class_time_sum[cls_id_str] += elapsed
        class_instance_count[cls_id_str] += 1
        image_time_sum[img_key] += elapsed

    # ================= 数据汇总与输出 =================
    unique_images_count = len(image_time_sum)
    avg_time_per_image = sum(image_time_sum.values()) / unique_images_count if unique_images_count > 0 else 0
    avg_time_overall = total_time / total_instances if total_instances > 0 else 0
    
    txt_file = os.path.join(save_dir, f'pure_inference_timing_{dataset_name}.txt')
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Strict Single-Instance Inference Timing ({dataset_name})\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Unique Images Processed : {unique_images_count}\n")
        f.write(f"Total Instances Processed     : {total_instances}\n")
        f.write("-" * 60 + "\n")
        
        f.write(f"[1] 所有类别的每个实例预测所需平均时间 : {avg_time_overall:.5f} s ({avg_time_overall*1000:.2f} ms)\n")
        f.write(f"[2] 每张图像预测所需平均时间         : {avg_time_per_image:.5f} s ({avg_time_per_image*1000:.2f} ms)\n")
        
        f.write("-" * 60 + "\n")
        f.write("[3] 每个类别的每个实例预测所需平均时间 :\n")
        for cls_id in sorted([int(k) for k in class_instance_count.keys()]):
            cls_str = str(cls_id)
            count = class_instance_count[cls_str]
            avg_time = class_time_sum[cls_str] / count if count > 0 else 0
            f.write(f"    - Obj ID: {cls_id:02d} | Count: {count:<5} | Avg Time: {avg_time:.5f} s ({avg_time*1000:.2f} ms)\n")
        
        f.write("\n" + "=" * 60 + "\n")

    print(f"\n✅ 评测完成！详细时间统计已保存至:\n{txt_file}")