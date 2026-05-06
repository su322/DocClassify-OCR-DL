"""
检查现有缓存，计算验证集可以分离多少

用法:
    python training/scripts/check_val_split.py
"""

import os
import sys
import pickle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def main():
    # 路径配置
    cache_file = "training/data/rvl_cdip/cache/processed_data.pkl"
    val_dir = "training/data/rvl_cdip/val"

    if not os.path.exists(cache_file):
        print(f"错误: 找不到 {cache_file}")
        return

    # 加载缓存
    print(f"加载 {cache_file}...")
    with open(cache_file, "rb") as f:
        cached = pickle.load(f)

    dataset = cached.get("dataset", [])
    processed_files = set(cached.get("processed_files", []))
    print(f"共 {len(dataset)} 个样本, {len(processed_files)} 个文件记录")
    print(f"失败/跳过: {len(processed_files) - len(dataset)} 个文件")

    # 获取验证集文件列表
    val_files = set()
    if os.path.isdir(val_dir):
        for class_name in os.listdir(val_dir):
            class_dir = os.path.join(val_dir, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pdf", ".bmp")):
                        val_files.add(os.path.join(class_name, f))
    print(f"\n验证集目录: {len(val_files)} 个文件")

    # 检查验证集文件有多少已处理
    val_processed = val_files & processed_files
    val_not_processed = val_files - processed_files
    print(f"  已处理: {len(val_processed)} 个")
    print(f"  未处理: {len(val_not_processed)} 个")

    # 估算：假设成功率相同
    success_rate = len(dataset) / len(processed_files) if processed_files else 0
    estimated_val_samples = int(len(val_processed) * success_rate)
    print(f"\n成功率: {success_rate:.2%}")
    print(f"预估验证集样本数: {estimated_val_samples}")

    # 训练集
    train_dir = "training/data/rvl_cdip/train"
    train_files = set()
    if os.path.isdir(train_dir):
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pdf", ".bmp")):
                        train_files.add(os.path.join(class_name, f))

    train_processed = train_files & processed_files
    estimated_train_samples = int(len(train_processed) * success_rate)
    print(f"\n训练集目录: {len(train_files)} 个文件")
    print(f"  已处理: {len(train_processed)} 个")
    print(f"预估训练集样本数: {estimated_train_samples}")

    print(f"\n预估总计: {estimated_train_samples + estimated_val_samples} (实际 {len(dataset)})")


if __name__ == "__main__":
    main()
