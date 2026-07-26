import argparse
import pandas as pd
from HccePose.bop_loader import BopDataset

def count_instances(dataset_root, train_folder, test_folder, train_vis_threshold=0.2, test_vis_threshold=0.2):
    print("=" * 60)
    print(f"Dataset Root: {dataset_root}")
    print(f"Train Folder: {train_folder} (vis >= {train_vis_threshold})")
    print(f"Test Folder:  {test_folder} (vis >= {test_vis_threshold})")
    print("=" * 60)

    # 1. 实例化基础加载器
    bop_dataset = BopDataset(dataset_root, model_name='models', local_rank=0)
    if not bop_dataset.obj_id_list:
        print("未找到物体模型信息，请检查 dataset_root 和 model_name。")
        return

    # 获取数据集中所有的物体 ID 并排序
    all_obj_ids = sorted(bop_dataset.obj_id_list)

    # 2. 统计训练集实例
    print(f"\n[1/2] 正在扫描训练集: {train_folder} ...")
    # scene_num 设大一点以确保扫描所有场景文件夹
    train_data = bop_dataset.load_folder(train_folder, scene_num=5000, vis=train_vis_threshold)
    
    train_counts = {obj_id: 0 for obj_id in all_obj_ids}
    if train_data is not None:
        obj_info = train_data['obj_info']
        for obj_key, instances in obj_info.items():
            # obj_key 的格式为 'obj_000001'
            obj_id = int(obj_key.split('_')[1])
            train_counts[obj_id] = len(instances)

    # 3. 统计测试集实例
    print(f"\n[2/2] 正在扫描测试集: {test_folder} ...")
    test_data = bop_dataset.load_folder(test_folder, scene_num=5000, vis=test_vis_threshold)
    
    test_counts = {obj_id: 0 for obj_id in all_obj_ids}
    if test_data is not None:
        obj_info = test_data['obj_info']
        for obj_key, instances in obj_info.items():
            obj_id = int(obj_key.split('_')[1])
            test_counts[obj_id] = len(instances)

    # 4. 汇总数据与输出格式化
    data_list = []
    total_train = 0
    total_test = 0

    for obj_id in all_obj_ids:
        tr_count = train_counts[obj_id]
        te_count = test_counts[obj_id]
        total_train += tr_count
        total_test += te_count
        
        data_list.append({
            "Obj_ID": obj_id,
            "Train_Instances": tr_count,
            "Test_Instances": te_count,
            "Ratio (Train/Test)": f"{tr_count/te_count:.2f}" if te_count > 0 else "N/A"
        })

    df = pd.DataFrame(data_list)
    
    print("\n" + "=" * 60)
    print("                      DATASET STATISTICS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    print(f"OVERALL TOTAL | Train: {total_train:<8} | Test: {total_test:<8}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count BOP dataset instances per class")
    parser.add_argument('--dataset_root', type=str, required=True, 
                        help="Path to dataset root (e.g., /media/ubuntu/.../tless)")
    parser.add_argument('--train_folder', type=str, default='train_pbr', 
                        help="Name of the training folder")
    parser.add_argument('--test_folder', type=str, default='test_primesense', 
                        help="Name of the testing folder")
    parser.add_argument('--train_vis', type=float, default=0.2, 
                        help="Visibility threshold for training")
    parser.add_argument('--test_vis', type=float, default=0.2, 
                        help="Visibility threshold for testing")
    args = parser.parse_args()

    count_instances(
        dataset_root=args.dataset_root,
        train_folder=args.train_folder,
        test_folder=args.test_folder,
        train_vis_threshold=args.train_vis,
        test_vis_threshold=args.test_vis
    )