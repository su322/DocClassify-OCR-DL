"""
训练流程快速验证脚本

前提: 已运行 prepare_rvl_cdip.py 准备好数据集
用途: 用最少的数据快速验证 CNN/GCN/GAT 训练流程是否有 bug
      从已有训练数据中每个类别取 3 张图（共 ~48 张），几分钟跑完

用法:
    python training/scripts/test_train.py
"""

import os
import sys
import shutil

# 添加项目根目录到 Python 路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.core.config import settings

# ============================================================
# 配置
# ============================================================
DATASET = "rvl_cdip"
SAMPLES_PER_CLASS = 3  # 每个类别取几张
TEST_DIR = "training/data/rvl_cdip/_test_small"  # 临时小数据集目录


def create_small_dataset():
    """从训练集中每个类别取少量样本，创建小数据集"""
    ds = settings.DATASETS[DATASET]
    train_dir = ds["train_dir"]
    classes = ds["classes"]

    # 清理旧的临时目录
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    os.makedirs(TEST_DIR, exist_ok=True)

    total = 0
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [跳过] {cls} 目录不存在")
            continue

        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ][:SAMPLES_PER_CLASS]

        target_cls_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(target_cls_dir, exist_ok=True)

        for f in files:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(target_cls_dir, f))
            total += 1

        print(f"  [{cls}] 取了 {len(files)} 张")
    print(f"\n  共复制 {total} 张到 {TEST_DIR}/\n")
    return total


def test_cnn():
    """测试 CNN 训练（无独立验证集时自动用训练集代替）"""
    print("=" * 60)
    print("  测试 1: CNN 训练 (3 epochs)")
    print("  注意: 临时目录无 val 文件夹，将用训练集代替验证集")
    print("=" * 60)

    from training.scripts.train_cnn import train_cnn

    ds = settings.DATASETS[DATASET]
    classes = ds["classes"]

    try:
        train_cnn(
            data_dir=TEST_DIR,
            classes=classes,
            val_dir=None,
            test_dir=None,
            epochs=3,
            batch_size=4,
            learning_rate=0.001,
            patience=10,
            output_dir="training/output/_test_cnn",
        )
        print("\n  ✅ CNN 训练通过\n")
        return True
    except Exception as e:
        print(f"\n  ❌ CNN 训练失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_prepare():
    """测试 GNN 数据准备（OCR + 图构建）"""
    print("=" * 60)
    print("  测试 2: GNN 数据准备 (OCR + 图构建)")
    print("=" * 60)

    from training.scripts.train_gnn import prepare_train_data

    ds = settings.DATASETS[DATASET]
    classes = ds["classes"]

    try:
        dataset = prepare_train_data(
            data_dir=TEST_DIR,
            classes=classes,
            cache_dir=None,  # 不用缓存，直接测试
        )
        print(f"\n  共生成 {len(dataset)} 个图数据")

        if len(dataset) == 0:
            print("  ❌ 没有生成任何图数据！")
            return False

        # 检查数据格式
        sample = dataset[0]
        print(f"  节点特征维度: {sample.x.shape}")
        print(f"  边索引形状: {sample.edge_index.shape}")
        print(f"  标签: {sample.y.item()}")
        assert sample.x.shape[1] == 398, f"特征维度应为 398，实际为 {sample.x.shape[1]}"
        print("\n  ✅ GNN 数据准备通过\n")
        return True
    except Exception as e:
        print(f"\n  ❌ GNN 数据准备失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gnn_train():
    """测试 GNN 训练"""
    print("=" * 60)
    print("  测试 3: GNN 训练 (GCN + GAT, 5 epochs)")
    print("=" * 60)

    from training.scripts.train_gnn import prepare_train_data, train_model

    ds = settings.DATASETS[DATASET]
    classes = ds["classes"]

    try:
        dataset = prepare_train_data(
            data_dir=TEST_DIR,
            classes=classes,
            cache_dir=None,
        )

        if len(dataset) == 0:
            print("  ❌ 没有图数据，跳过训练")
            return False

        # 划分一小部分当验证集
        split = max(1, len(dataset) // 3)
        train_data = dataset[:-split]
        val_data = dataset[-split:]

        for model_name in ["gcn", "gat"]:
            print(f"\n  --- {model_name.upper()} ---")
            history = train_model(
                model_name=model_name,
                dataset=train_data,
                val_dataset=val_data,
                classes=classes,
                epochs=5,
                batch_size=4,
                learning_rate=0.001,
                patience=10,
                output_dir=f"training/output/_test_{model_name}",
            )
            print(f"  {model_name.upper()} 最佳验证准确率: {max(history['val_acc']):.4f}")
            print(f"  ✅ {model_name.upper()} 训练通过")

        print("\n  ✅ GNN 训练全部通过\n")
        return True
    except Exception as e:
        print(f"\n  ❌ GNN 训练失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """清理临时文件"""
    print("清理临时文件...")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    for d in ["training/output/_test_cnn", "training/output/_test_gcn", "training/output/_test_gat"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("清理完成\n")


def main():
    print("\n" + "=" * 60)
    print("  GNN 训练流程验证")
    print("=" * 60 + "\n")

    # 检查数据目录
    ds = settings.DATASETS[DATASET]
    if not os.path.isdir(ds["train_dir"]):
        print(f"错误: 训练数据目录不存在: {ds['train_dir']}")
        print("请先运行 prepare_rvl_cdip.py 准备数据")
        return

    # 创建小数据集
    print("创建小数据集...\n")
    create_small_dataset()

    # 运行测试
    results = {}
    results["CNN 训练"] = test_cnn()
    results["GNN 数据准备"] = test_gnn_prepare()
    results["GNN 训练"] = test_gnn_train()

    # 汇总
    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    # 清理
    cleanup()

    if all_pass:
        print("\n🎉 全部测试通过！可以放心跑全量训练。\n")
    else:
        print("\n⚠️  有测试失败，请检查上面的错误信息。\n")


if __name__ == "__main__":
    main()
