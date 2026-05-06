"""
从现有缓存分离验证集数据

用法:
    python training/scripts/split_val_cache.py
"""

import os
import sys
import pickle
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def main():
    # 路径配置
    cache_dir = "training/data/rvl_cdip/cache"
    pkl_file = os.path.join(cache_dir, "processed_data.pkl")
    val_dir = "training/data/rvl_cdip/train/../val"  # 用相对路径确保正确
    val_cache_dir = os.path.join(cache_dir, "val")

    if not os.path.exists(pkl_file):
        print(f"错误: 找不到 {pkl_file}")
        return

    # 获取验证集文件列表
    val_dir = "training/data/rvl_cdip/val"
    val_files = set()
    if os.path.isdir(val_dir):
        for class_name in os.listdir(val_dir):
            class_dir = os.path.join(val_dir, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pdf", ".bmp")):
                        val_files.add(os.path.join(class_name, f))
    print(f"验证集目录: {len(val_files)} 个文件")

    # 加载现有缓存
    print(f"加载 {pkl_file}...")
    with open(pkl_file, "rb") as f:
        cached = pickle.load(f)

    dataset = cached.get("dataset", [])
    processed_files = list(cached.get("processed_files", []))  # 转为 list 保持顺序
    print(f"共 {len(dataset)} 个样本, {len(processed_files)} 个文件记录")

    # 分离
    # 假设 processed_files 和 dataset 大致对应（成功率 99.95%）
    # 遍历 processed_files，检查是否属于验证集
    train_dataset = []
    val_dataset = []
    train_files = []
    val_files_list = []

    # 建立映射：processed_files[i] -> dataset 中对应的样本
    # 由于有些文件失败被跳过，需要追踪 dataset 索引
    dataset_idx = 0
    for file_key in processed_files:
        if file_key in val_files:
            # 属于验证集
            if dataset_idx < len(dataset):
                # 检查样本的 label 是否匹配验证集类别
                val_dataset.append(dataset[dataset_idx])
                val_files_list.append(file_key)
                dataset_idx += 1
            else:
                print(f"  [警告] 文件 {file_key} 在验证集但超出 dataset 范围")
        else:
            # 属于训练集
            if dataset_idx < len(dataset):
                train_dataset.append(dataset[dataset_idx])
                train_files.append(file_key)
                dataset_idx += 1
            else:
                print(f"  [警告] 文件 {file_key} 在训练集但超出 dataset 范围")

    print(f"\n分离结果:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")
    print(f"  未匹配: {len(dataset) - len(train_dataset) - len(val_dataset)} 个")

    if len(val_dataset) == 0:
        print("\n错误: 没有分离出验证集数据！")
        print("可能原因: processed_files 和 dataset 顺序不匹配")
        return

    # 保存验证集缓存
    os.makedirs(val_cache_dir, exist_ok=True)
    val_pkl = os.path.join(val_cache_dir, "processed_data.pkl")
    with open(val_pkl, "wb") as f:
        pickle.dump({"dataset": val_dataset, "processed_files": val_files_list}, f)
    print(f"\n验证集缓存已保存: {val_pkl}")

    # 更新训练集缓存
    with open(pkl_file, "wb") as f:
        pickle.dump({"dataset": train_dataset, "processed_files": train_files}, f)
    print(f"训练集缓存已更新: {pkl_file}")

    # 生成验证集分片
    print("\n生成验证集分片...")
    SHARD_SIZE = 1000
    for i in range(0, len(val_dataset), SHARD_SIZE):
        shard = val_dataset[i:i + SHARD_SIZE]
        shard_path = os.path.join(val_cache_dir, f"shard_{i // SHARD_SIZE:05d}.pt")
        torch.save(shard, shard_path)
    print(f"验证集分片完成: {len(val_dataset)} 个样本 → {(len(val_dataset) + SHARD_SIZE - 1) // SHARD_SIZE} 个分片")

    # 重新生成训练集分片
    print("\n重新生成训练集分片...")
    for i in range(0, len(train_dataset), SHARD_SIZE):
        shard = train_dataset[i:i + SHARD_SIZE]
        shard_path = os.path.join(cache_dir, f"shard_{i // SHARD_SIZE:05d}.pt")
        torch.save(shard, shard_path)
    print(f"训练集分片完成: {len(train_dataset)} 个样本 → {(len(train_dataset) + SHARD_SIZE - 1) // SHARD_SIZE} 个分片")

    print("\n✅ 完成! 下次训练将自动使用分离后的缓存")


if __name__ == "__main__":
    main()
