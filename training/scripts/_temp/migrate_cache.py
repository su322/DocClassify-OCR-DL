"""
缓存迁移脚本：将整体 pkl 转为分片 .pt 格式

用法:
    python training/scripts/migrate_cache.py

前提: cache/processed_data.pkl 存在
结果: 生成 cache/shard_XXXXX.pt + cache/processed_files.json
"""

import os
import sys
import json
import pickle
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

SHARD_SIZE = 1000  # 每个分片 1000 个图


def migrate(cache_dir: str):
    pkl_path = os.path.join(cache_dir, "processed_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"错误: 找不到 {pkl_path}")
        return

    print(f"加载 {pkl_path}...")
    with open(pkl_path, "rb") as f:
        cached = pickle.load(f)

    dataset = cached.get("dataset", [])
    processed_files = cached.get("processed_files", [])
    print(f"共 {len(dataset)} 个样本, {len(processed_files)} 个文件记录")

    # 保存 processed_files.json
    json_path = os.path.join(cache_dir, "processed_files.json")
    with open(json_path, "w") as f:
        json.dump(processed_files, f)
    print(f"已保存: {json_path}")

    # 分片保存
    total = len(dataset)
    shard_count = (total + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"分片: {shard_count} 个文件, 每片 {SHARD_SIZE} 个")

    for i in range(shard_count):
        start = i * SHARD_SIZE
        end = min((i + 1) * SHARD_SIZE, total)
        shard = dataset[start:end]
        shard_path = os.path.join(cache_dir, f"shard_{i:05d}.pt")
        torch.save(shard, shard_path)
        print(f"  [{i+1}/{shard_count}] {shard_path} ({len(shard)} 个样本)")

    print(f"\n迁移完成! 共 {shard_count} 个分片")
    print(f"旧文件 {pkl_path} 可以手动删除（建议先确认训练正常）")


def main():
    # 默认缓存目录
    cache_dir = "training/data/rvl_cdip/cache"
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]

    if not os.path.isdir(cache_dir):
        print(f"错误: 目录不存在: {cache_dir}")
        return

    migrate(cache_dir)


if __name__ == "__main__":
    main()
