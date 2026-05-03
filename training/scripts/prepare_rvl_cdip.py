"""
RVL-CDIP 数据集预处理脚本

用法:
    1. 下载 rvl-cdip.tar.gz 和 train.txt / val.txt / test.txt
    2. 解压: tar -xzf rvl-cdip.tar.gz
    3. 运行此脚本: python training/scripts/prepare_rvl_cdip.py

脚本会自动:
    - 将数字编号文件夹重命名为类别名
    - 按 train/val/test 划分复制到对应目录
    - 每个类别可限制最大样本数（默认 1000，防止数据量过大）
"""

import os
import shutil

# RVL-CDIP 数字编号 → 类别名映射
LABEL_MAP = {
    "0": "letter",
    "1": "form",
    "2": "email",
    "3": "handwritten",
    "4": "advertisement",
    "5": "scientific_report",
    "6": "scientific_publication",
    "7": "specification",
    "8": "file_folder",
    "9": "news_article",
    "10": "budget",
    "11": "invoice",
    "12": "presentation",
    "13": "questionnaire",
    "14": "resume",
    "15": "memo",
}

# 配置
RVL_CDIP_ROOT = "training/data/rvl_cdip"  # 项目中 RVL-CDIP 的根目录
EXTRACTED_DIR = "training/data/rvl_cdip/raw"  # 解压后的原始数据目录（数字编号文件夹）
MAX_SAMPLES_PER_CLASS = 1000  # 每个类别最大样本数（设为 None 表示不限制）


def parse_split_file(split_file: str) -> dict:
    """
    解析 train.txt / val.txt / test.txt

    文件格式: 每行 "图像路径 类别编号"
    例如: train/0/00001.png 0
    """
    samples = {}  # {label_name: [file_path, ...]}
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            img_path, label_id = parts
            label_name = LABEL_MAP.get(label_id)
            if label_name is None:
                continue
            if label_name not in samples:
                samples[label_name] = []
            samples[label_name].append(img_path)

    return samples


def main():
    # 检查解压目录
    if not os.path.isdir(EXTRACTED_DIR):
        print(f"错误: 找不到解压目录 {EXTRACTED_DIR}")
        print(f"请先解压 rvl-cdip.tar.gz 到 {EXTRACTED_DIR}/")
        return

    # 检查标签文件
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(RVL_CDIP_ROOT, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"错误: 找不到 {split_file}")
            print("请从 HuggingFace 下载 train.txt / val.txt / test.txt")
            return

    # 合并 train + val 作为训练集（我们的脚本会自动再划分验证集）
    print("解析标签文件...")
    all_samples = {}
    for split_name in ["train", "val"]:
        split_file = os.path.join(RVL_CDIP_ROOT, f"{split_name}.txt")
        samples = parse_split_file(split_file)
        for label, paths in samples.items():
            if label not in all_samples:
                all_samples[label] = []
            all_samples[label].extend(paths)

    # 统计
    print(f"\n共 {len(all_samples)} 个类别:")
    total = 0
    for label, paths in sorted(all_samples.items()):
        count = len(paths)
        if MAX_SAMPLES_PER_CLASS and count > MAX_SAMPLES_PER_CLASS:
            count = MAX_SAMPLES_PER_CLASS
        print(f"  {label}: {count} 张")
        total += count
    print(f"  总计: {total} 张\n")

    # 创建目标目录并复制文件
    target_dir = os.path.join(RVL_CDIP_ROOT, "train")
    os.makedirs(target_dir, exist_ok=True)

    for label, paths in sorted(all_samples.items()):
        label_dir = os.path.join(target_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # 限制样本数
        if MAX_SAMPLES_PER_CLASS:
            paths = paths[:MAX_SAMPLES_PER_CLASS]

        copied = 0
        for img_rel_path in paths:
            # 原始文件路径（相对于解压目录）
            src = os.path.join(EXTRACTED_DIR, img_rel_path)
            if not os.path.exists(src):
                continue

            # 目标文件路径
            filename = os.path.basename(img_rel_path)
            dst = os.path.join(label_dir, filename)

            # 复制文件（如果已存在则跳过）
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1

        print(f"  [{label}] 复制 {copied}/{len(paths)} 张")

    print(f"\n完成! 训练数据已准备到: {target_dir}")
    print(
        f"运行训练: python training/scripts/train_gnn.py --dataset rvl_cdip --model both"
    )


if __name__ == "__main__":
    main()
