"""
RVL-CDIP 数据集预处理脚本（适用于没有划分的情况）

功能:
- 从 source_dir（原始数据，文件夹名带空格）读取图像
- 按 80/10/10 划分为 train/val/test
- 输出到 train_dir/val_dir/test_dir（文件夹名用下划线，与代码类别名一致）
- 完成后删除原始 data/ 目录

用法:
    python training/scripts/prepare_rvl_cdip.py

注意：
    由于是随机划分的，在执行train_gnn的过程中，如果数据集划分不同的话，产生的缓存内容在每一次都是不同的，所以如果在其他机器上重新划分了，会有不同的地方需要重新处理，但大部分可能还是有的
"""

import os
import random
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="RVL-CDIP 数据集预处理")
    parser.add_argument("--dataset", type=str, default="rvl_cdip")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.dataset not in settings.DATASETS:
        print(f"错误: 未知数据集 '{args.dataset}'")
        sys.exit(1)

    ds = settings.DATASETS[args.dataset]
    source_dir = ds["source_dir"]
    train_dir = ds["train_dir"]
    val_dir = ds["val_dir"]
    test_dir = ds["test_dir"]
    classes = ds["classes"]

    # 文件夹名 → 类别名映射（数据集文件夹名带空格，代码中用下划线）
    folder_map = {cls.replace("_", " "): cls for cls in classes}

    if not os.path.isdir(source_dir):
        print(f"错误: 源数据目录不存在: {source_dir}")
        sys.exit(1)

    random.seed(args.seed)

    # 扫描源目录
    print(f"源数据目录: {source_dir}\n")

    class_files = {}  # {class_name: [filepath, ...]}
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        class_name = folder_map.get(folder_name)
        if class_name is None:
            print(f"  [跳过] 未知文件夹: {folder_name}")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        class_files[class_name] = [os.path.join(folder_path, f) for f in files]

    # 统计
    total = 0
    for cls in classes:
        count = len(class_files.get(cls, []))
        print(f"  {cls}: {count} 张")
        total += count
    print(f"  总计: {total} 张\n")

    # 按类别内 80/10/10 划分
    splits = {"train": {}, "val": {}, "test": {}}
    train_total, val_total, test_total = 0, 0, 0

    for cls in classes:
        files = class_files.get(cls, [])
        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        splits["train"][cls] = files[:n_train]
        splits["val"][cls] = files[n_train:n_train + n_val]
        splits["test"][cls] = files[n_train + n_val:]

        train_total += len(splits["train"][cls])
        val_total += len(splits["val"][cls])
        test_total += len(splits["test"][cls])

    print(f"划分结果:")
    print(f"  训练集: {train_total} 张 ({train_total/total:.1%})")
    print(f"  验证集: {val_total} 张 ({val_total/total:.1%})")
    print(f"  测试集: {test_total} 张 ({test_total/total:.1%})")
    print(f"  总计:   {total} 张\n")

    # 移动文件到对应目录（完成后删除原始 data/ 目录）
    import shutil
    split_dirs = {"train": train_dir, "val": val_dir, "test": test_dir}

    for split_name, target_root in split_dirs.items():
        os.makedirs(target_root, exist_ok=True)

        for cls, files in splits[split_name].items():
            cls_dir = os.path.join(target_root, cls)
            os.makedirs(cls_dir, exist_ok=True)

            for src_path in files:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(cls_dir, filename)
                shutil.move(src_path, dst_path)

        print(f"  [{split_name}] → {target_root}/ ({sum(len(v) for v in splits[split_name].values())} 张)")

    # 删除原始 data/ 目录（已全部移出）
    try:
        shutil.rmtree(source_dir)
        print(f"\n  已删除原始数据目录: {source_dir}")
    except OSError as e:
        print(f"\n  [警告] 删除原始目录失败: {e}，请手动删除")

    print(f"\n完成!")
    print(f"  训练: {train_dir}")
    print(f"  验证: {val_dir}")
    print(f"  测试: {test_dir}")
    print(f"\n运行训练:")
    print(f"  python training/scripts/train_cnn.py --dataset rvl_cdip")
    print(f"  python training/scripts/train_gnn.py --dataset rvl_cdip --model both")


if __name__ == "__main__":
    main()
