"""
GNN 文档分类模型训练脚本

功能:
- 支持 GCN / GAT 模型切换与对比实验
- 使用 PyG DataLoader 进行 mini-batch 训练
- 训练集/验证集自动划分
- 学习率调度 (ReduceLROnPlateau) + 早停机制
- 自动生成训练可视化（loss/acc 曲线、混淆矩阵、分类报告）
- 传递文档尺寸和版面类型信息用于特征构建
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.models.deep_learning.gnn_model import DocumentGNN
from backend.models.deep_learning.gat_model import DocumentGAT
from backend.services.classification_service import ClassificationService
from backend.services.ocr_service import ocr_service
from backend.core.config import settings

# ============================================================
# 工具函数
# ============================================================


def set_seed(seed: int = 42):
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count() or 4)  # GNN 没有 DataLoader workers，用满 CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(
    model_name: str, in_channels: int, hidden_channels: int, out_channels: int
):
    """
    根据名称创建模型

    Args:
        model_name: "gcn" 或 "gat"
        in_channels: 输入特征维度
        hidden_channels: 隐藏层维度
        out_channels: 输出类别数
    """
    if model_name == "gat":
        return DocumentGAT(in_channels, hidden_channels, out_channels, heads=4)
    elif model_name == "gcn":
        return DocumentGNN(in_channels, hidden_channels, out_channels)
    else:
        raise ValueError(f"未知模型: {model_name}，可选: 'gcn', 'gat'")


# ============================================================
# 数据准备
# ============================================================


class GraphDataset(Dataset):
    """
    按需加载分片 .pt 文件的图数据集
    内存占用从 ~21GB 降到 ~500MB
    """

    def __init__(self, cache_dir: str, shard_cache_size: int = 3):
        super().__init__(root=cache_dir)
        self.cache_dir = cache_dir
        self.shard_paths = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ])
        # 计算每个分片的样本数
        self.shard_lengths = []
        for p in self.shard_paths:
            shard = torch.load(p, weights_only=False)
            self.shard_lengths.append(len(shard))
        self.total = sum(self.shard_lengths)
        # 构建索引映射: 全局索引 -> (分片索引, 分片内索引)
        self._index_map = []
        for shard_idx, length in enumerate(self.shard_lengths):
            for local_idx in range(length):
                self._index_map.append((shard_idx, local_idx))
        # LRU 分片缓存：保留最近访问的几个分片，避免重复加载
        from collections import OrderedDict
        self._shard_cache = OrderedDict()  # shard_idx -> shard_data
        self._shard_cache_size = shard_cache_size

    def _load_shard(self, shard_idx: int):
        """加载分片，优先从缓存读取"""
        if shard_idx in self._shard_cache:
            # 移到末尾（最近使用）
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]
        # 缓存未命中，从磁盘加载
        shard = torch.load(self.shard_paths[shard_idx], weights_only=False)
        self._shard_cache[shard_idx] = shard
        # 超出缓存大小，淘汰最久未使用的
        while len(self._shard_cache) > self._shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def len(self):
        return self.total

    def get(self, idx):
        shard_idx, local_idx = self._index_map[idx]
        shard = self._load_shard(shard_idx)
        return shard[local_idx]


def prepare_train_data(data_dir: str, classes: list, cache_dir: str = None, save_interval: int = 100):
    """
    准备训练数据，将每个文档转换为 PyG Data 对象
    支持断点续跑：每 save_interval 张保存一次，中断后可从缓存恢复

    Args:
        data_dir: 训练数据目录
        classes: 类别列表（决定 class_to_idx 映射）
        cache_dir: 缓存目录，默认 data_dir/../cache
        save_interval: 每多少张保存一次
    """
    import pickle

    data_type = "验证集" if "val" in data_dir else "训练集"

    classification_service = ClassificationService(document_classes=classes)

    # 设置缓存目录
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(data_dir), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 从缓存加载已处理的数据
    cache_file = os.path.join(cache_dir, "processed_data.pkl")
    processed_files = set()
    dataset = []

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                dataset = cached_data.get("dataset", [])
                processed_files = set(cached_data.get("processed_files", []))
            print(f"  从缓存恢复: {len(dataset)} 个样本, {len(processed_files)} 个文件已处理")
        except Exception as e:
            print(f"  缓存加载失败: {e}，重新开始")
            dataset = []
            processed_files = set()
    else:
        print(f"\n{'='*60}")
        print(f"  未检测到缓存，开始准备 {data_type} 数据")
        print(f"  数据目录: {data_dir}")
        print(f"{'='*60}")

    total_new = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in classification_service.class_to_idx:
            print(f"  [跳过] 未知类别: {class_name}")
            continue
        label = classification_service.class_to_idx[class_name]

        file_list = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        ]
        file_list.sort()

        # 过滤掉已处理的文件
        pending_files = [f for f in file_list if os.path.join(class_name, f) not in processed_files]
        if len(pending_files) < len(file_list):
            print(f"  [{class_name}] 找到 {len(file_list)} 个文件, {len(pending_files)} 个待处理")
        else:
            print(f"  [{class_name}] 找到 {len(file_list)} 个文件")

        for idx, filename in enumerate(pending_files):
            file_path = os.path.join(class_dir, filename)
            file_key = os.path.join(class_name, filename)
            print(f"    处理: {filename}...", end=" ")

            try:
                doc_id = f"train_{class_name}_{filename}"

                # macOS 上 PaddleOCR 可能无法直接处理 .tif，转为 .png 兼容
                actual_file_path = file_path
                tmp_png = None
                if filename.lower().endswith((".tif", ".tiff")):
                    from PIL import Image
                    import tempfile
                    img = Image.open(file_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    img.save(tmp_png.name)
                    tmp_png.close()
                    actual_file_path = tmp_png.name

                ocr_result = ocr_service.process_document(actual_file_path, filename, doc_id)

                # 清理临时文件
                if tmp_png and os.path.exists(tmp_png.name):
                    os.unlink(tmp_png.name)

                doc_width = float(ocr_result.width) if ocr_result.width else 0
                doc_height = float(ocr_result.height) if ocr_result.height else 0

                graph = classification_service._build_graph(
                    ocr_result.regions,
                    ocr_result.tables,
                    doc_width=doc_width,
                    doc_height=doc_height,
                )

                if graph["num_nodes"] == 0:
                    print("跳过（无有效节点）")
                    processed_files.add(file_key)
                    continue

                data = Data(
                    x=torch.tensor(graph["node_features"], dtype=torch.float32),
                    edge_index=torch.tensor(graph["edges"], dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.long),
                )
                dataset.append(data)
                processed_files.add(file_key)
                total_new += 1
                print(
                    f"OK [{len(dataset)}] (节点: {graph['num_nodes']}, 边: {graph['edges'].shape[1] if graph['edges'].size > 0 else 0})"
                )

                # 每 save_interval 张保存一次缓存
                if total_new % save_interval == 0:
                    with open(cache_file, "wb") as f:
                        pickle.dump({"dataset": dataset, "processed_files": list(processed_files)}, f)
                    print(f"  [自动保存] 已处理 {len(dataset)} 个样本")

            except Exception as e:
                print(f"失败: {e}")

    # 最终保存
    if total_new > 0:
        with open(cache_file, "wb") as f:
            pickle.dump({"dataset": dataset, "processed_files": list(processed_files)}, f)
        print(f"  [最终保存] 共 {len(dataset)} 个样本")

    return dataset


# ============================================================
# 训练与评估
# ============================================================


def evaluate(model, loader, criterion, device):
    """在给定 DataLoader 上评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            if batch.num_graphs == 0 or batch.y.numel() == 0:
                continue
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.squeeze(-1))
            total_loss += loss.item() * batch.num_graphs
            _, predicted = torch.max(output, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.squeeze(-1).cpu().numpy())
            total += batch.num_graphs

    avg_loss = total_loss / total
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model_name: str,
    dataset: list,
    val_dataset: list = None,
    classes: list = None,
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    patience: int = 20,
    output_dir: str = "training/output"
):
    """
    训练指定模型并生成可视化结果

    Args:
        model_name: 模型名称 ("gcn" 或 "gat")
        dataset: PyG Data 对象列表
        classes: 类别列表（用于可视化标签）
        epochs: 最大训练轮数
        batch_size: 批大小
        learning_rate: 初始学习率
        val_ratio: 验证集比例
        patience: 早停耐心值
        output_dir: 输出目录（保存模型和可视化图表）
    """
    if not dataset:
        print("错误: 没有找到训练数据!")
        return

    class_names = classes if classes else settings.DOCUMENT_CLASSES
    train_start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 创建输出目录
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"训练配置 [{model_name.upper()}]")
    print(f"{'='*60}")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批大小: {batch_size}")
    print(f"  初始学习率: {learning_rate}")
    print(f"  早停耐心值: {patience}")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")

    # 使用预处理好的验证集
    if val_dataset:
        print(f"训练集: {len(dataset)} 样本, 验证集: {len(val_dataset)} 样本\n")
    else:
        print(f"训练集: {len(dataset)} 样本, 验证集: 无\n")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True) if val_dataset else None

    # 确定模型参数
    sample = dataset[0]
    in_channels = sample.x.shape[1]
    hidden_channels = 128
    out_channels = len(class_names)

    print(
        f"模型参数: in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}\n"
    )

    model = get_model(model_name, in_channels, hidden_channels, out_channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # 训练历史记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve_count = 0
    best_preds = None
    best_labels = None

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if batch.num_graphs == 0 or batch.y.numel() == 0:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y.squeeze(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            _, predicted = torch.max(output, dim=1)
            train_correct += (predicted == batch.y.squeeze(-1)).sum().item()
            train_total += batch.num_graphs

        avg_train_loss = train_loss / len(dataset)
        train_acc = train_correct / train_total

        # ---- 验证阶段 ----
        if val_loader:
            avg_val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
        else:
            avg_val_loss, val_acc, val_preds, val_labels = avg_train_loss, train_acc, [], []

        # 学习率调度
        if val_loader:
            scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录历史
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_preds = val_preds
            best_labels = val_labels
            no_improve_count = 0
            model_path = os.path.join(model_output_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"  → 保存最佳模型 (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f})"
            )
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= patience:
            print(f"\n早停触发: 验证集 loss 连续 {patience} 轮未改善。")
            break

    # ---- 训练结束 ----
    train_time = time.time() - train_start_time

    print(f"\n{'='*60}")
    print(f"训练完成! [{model_name.upper()}]")
    print(f"  最佳验证集 Loss: {best_val_loss:.4f}")
    print(f"  最佳验证集 Acc:  {best_val_acc:.4f}")
    print(f"  训练时间:        {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  模型参数量:      {total_params:,}")
    print(f"  训练轮数:        {len(history['train_loss'])}")
    print(f"{'='*60}\n")

    # 保存训练历史
    history_path = os.path.join(model_output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"训练历史已保存: {history_path}")

    # 生成可视化图表
    _generate_plots(
        history, best_preds, best_labels, model_name, model_output_dir, class_names
    )

    # 保存结果摘要（供对比脚本使用）
    result = {
        "model": model_name.upper(),
        "val_accuracy": round(best_val_acc, 4),
        "total_params": total_params,
        "train_time_seconds": round(train_time, 1),
        "train_epochs": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 4),
    }
    try:
        from sklearn.metrics import f1_score

        if best_preds and best_labels:
            result["val_macro_f1"] = round(
                f1_score(best_labels, best_preds, average="macro"), 4
            )
    except ImportError:
        pass

    result_path = os.path.join(model_output_dir, "result_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# 可视化
# ============================================================


def _generate_plots(
    history: dict,
    preds: list,
    true_labels: list,
    model_name: str,
    output_dir: str,
    class_names: list = None,
):
    """生成训练可视化图表"""
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无头模式，不弹出窗口
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        if class_names is None:
            class_names = settings.DOCUMENT_CLASSES

        # ---- 1. Loss 曲线 ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=1.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{model_name.upper()} - Loss Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ---- 2. Accuracy 曲线 ----
        ax2.plot(history["train_acc"], label="Train Acc", linewidth=1.5)
        ax2.plot(history["val_acc"], label="Val Acc", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{model_name.upper()} - Accuracy Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        loss_acc_path = os.path.join(output_dir, f"{model_name}_loss_acc.png")
        plt.savefig(loss_acc_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss/Acc 曲线已保存: {loss_acc_path}")

        # ---- 3. 混淆矩阵 ----
        if preds and true_labels:
            cm = confusion_matrix(true_labels, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"{model_name.upper()} - Confusion Matrix")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"混淆矩阵已保存: {cm_path}")

            # ---- 4. 分类报告 ----
            report = classification_report(
                true_labels, preds, target_names=class_names, digits=4,
                labels=list(range(len(class_names))), zero_division=0
            )
            report_path = os.path.join(
                output_dir, f"{model_name}_classification_report.txt"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"模型: {model_name.upper()}\n")
                f.write(f"{'='*60}\n\n")
                f.write(report)
            print(f"分类报告已保存: {report_path}")

    except ImportError:
        print("\n[警告] 缺少 matplotlib/sklearn/seaborn，跳过可视化生成。")
        print("请运行: pip install matplotlib scikit-learn seaborn")


# ============================================================
# 对比实验
# ============================================================


def run_comparison(dataset: list, val_dataset: list = None, **kwargs):
    """
    运行 GCN vs GAT 对比实验

    两个模型使用完全相同的数据划分和随机种子，确保公平对比。
    """
    print("\n" + "=" * 60)
    print("  对比实验: GCN vs GAT")
    print("=" * 60)

    results = {}
    for model_name in ["gcn", "gat"]:
        print(f"\n{'#'*60}")
        print(f"  开始训练: {model_name.upper()}")
        print(f"{'#'*60}\n")
        set_seed(42)  # 每个模型使用相同的种子，确保数据划分一致
        history = train_model(model_name, dataset, val_dataset=val_dataset, **kwargs)
        results[model_name] = {
            "best_val_loss": min(history["val_loss"]),
            "best_val_acc": max(history["val_acc"]),
            "total_epochs": len(history["train_loss"]),
        }

    # 打印对比结果
    print("\n" + "=" * 60)
    print("  对比实验结果")
    print("=" * 60)
    print(f"{'指标':<20} {'GCN':<15} {'GAT':<15}")
    print("-" * 50)
    print(
        f"{'最佳验证集 Loss':<20} {results['gcn']['best_val_loss']:<15.4f} {results['gat']['best_val_loss']:<15.4f}"
    )
    print(
        f"{'最佳验证集 Acc':<20} {results['gcn']['best_val_acc']:<15.4f} {results['gat']['best_val_acc']:<15.4f}"
    )
    print(
        f"{'训练轮数':<20} {results['gcn']['total_epochs']:<15} {results['gat']['total_epochs']:<15}"
    )
    print("=" * 60)

    # 保存对比结果
    output_dir = kwargs.get("output_dir", "training/output")
    comparison_path = os.path.join(output_dir, "comparison_results.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")

    return results


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN 文档分类模型训练")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["gcn", "gat", "both"],
        help="训练哪个模型: gcn / gat / both(对比实验)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="使用预定义数据集: rvl_cdip（与 --data-dir 二选一）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="自定义训练数据目录（与 --dataset 二选一）",
    )
    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument(
        "--output-dir", type=str, default="training/output", help="输出目录"
    )
    args = parser.parse_args()

    # 确定数据集和类别
    if args.dataset:
        if args.dataset not in settings.DATASETS:
            print(
                f"错误: 未知数据集 '{args.dataset}'，可选: {list(settings.DATASETS.keys())}"
            )
            sys.exit(1)
        ds_config = settings.DATASETS[args.dataset]
        data_dir = ds_config["train_dir"]
        classes = ds_config["classes"]
        print(f"使用预定义数据集: {args.dataset}")
        print(f"  描述: {ds_config['description']}")
        print(f"  类别数: {len(classes)}")
        print(f"  数据目录: {data_dir}")
    elif args.data_dir:
        data_dir = args.data_dir
        classes = settings.DOCUMENT_CLASSES
        print(f"使用自定义数据目录: {data_dir}")
    else:
        print("错误: 请指定 --dataset 或 --data-dir")
        print(f"  可用数据集: {list(settings.DATASETS.keys())}")
        sys.exit(1)

    set_seed(42)

    # 始终先准备数据（自动跳过已处理文件，处理新增文件）
    print("\n开始准备训练数据...")
    dataset = prepare_train_data(data_dir, classes)

    # 检查是否需要生成分片缓存（节省内存）
    default_cache_dir = os.path.join(os.path.dirname(data_dir), "cache")
    pkl_file = os.path.join(default_cache_dir, "processed_data.pkl")
    shard_files = [
        f for f in os.listdir(default_cache_dir)
        if f.startswith("shard_") and f.endswith(".pt")
    ] if os.path.isdir(default_cache_dir) else []

    # 如果没有分片，或 pkl 比分片更新，自动生成分片
    need_shard = False
    if not shard_files:
        need_shard = True
    elif os.path.exists(pkl_file):
        pkl_mtime = os.path.getmtime(pkl_file)
        shard_mtime = max(os.path.getmtime(os.path.join(default_cache_dir, f)) for f in shard_files)
        if pkl_mtime > shard_mtime:
            need_shard = True
            print("  检测到 pkl 已更新，重新生成分片...")

    if need_shard and os.path.exists(pkl_file):
        import pickle as _pkl
        print("  正在生成分片缓存...")
        # 先删除所有旧分片（避免残留）
        old_shards = [f for f in os.listdir(default_cache_dir) if f.startswith("shard_") and f.endswith(".pt")]
        for old_shard in old_shards:
            os.remove(os.path.join(default_cache_dir, old_shard))
        # 强制刷新文件系统缓存（macOS/Linux）
        os.sync()
        # 验证删除成功
        remaining = [f for f in os.listdir(default_cache_dir) if f.startswith("shard_") and f.endswith(".pt")]
        if remaining:
            print(f"  [警告] 仍有 {len(remaining)} 个分片未删除，强制删除...")
            for r in remaining:
                os.remove(os.path.join(default_cache_dir, r))
            os.sync()
        if old_shards:
            print(f"  已删除 {len(old_shards)} 个旧分片")
        with open(pkl_file, "rb") as f:
            cached = _pkl.load(f)
        all_data = cached.get("dataset", [])
        SHARD_SIZE = 1000
        for i in range(0, len(all_data), SHARD_SIZE):
            shard = all_data[i:i + SHARD_SIZE]
            shard_path = os.path.join(default_cache_dir, f"shard_{i // SHARD_SIZE:05d}.pt")
            torch.save(shard, shard_path)
        print(f"  分片完成: {len(all_data)} 个样本 → {(len(all_data) + SHARD_SIZE - 1) // SHARD_SIZE} 个分片")
        shard_files = [f for f in os.listdir(default_cache_dir) if f.startswith("shard_") and f.endswith(".pt")]

    if shard_files:
        dataset = GraphDataset(default_cache_dir)
        print(f"  使用按需加载模式: {len(dataset)} 个训练样本（内存占用低）")
    else:
        print(f"\n共准备 {len(dataset)} 个训练样本")

    # 准备验证集数据（如果存在 val 目录）
    val_dataset = None
    if args.dataset:
        val_dir = ds_config.get("val_dir")
    else:
        val_dir = None
    if val_dir and os.path.isdir(val_dir):
        # 验证集用独立缓存目录（避免和训练集混淆）
        val_cache_dir = os.path.join(os.path.dirname(val_dir), "cache", "val")
        val_shard_files = [
            f for f in os.listdir(val_cache_dir)
            if f.startswith("shard_") and f.endswith(".pt")
        ] if os.path.isdir(val_cache_dir) else []

        if val_shard_files:
            print(f"\n  检测到验证集分片缓存 ({len(val_shard_files)} 个分片)")
            val_dataset = GraphDataset(val_cache_dir)
            print(f"  共 {len(val_dataset)} 个验证样本（按需加载）")
        else:
            print("\n开始准备验证数据...")
            val_dataset = prepare_train_data(val_dir, classes, cache_dir=val_cache_dir)
            print(f"共准备 {len(val_dataset)} 个验证样本")

    print()

    train_kwargs = {
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "output_dir": args.output_dir,
    }

    if not dataset:
        print("错误: 没有有效的训练数据，请检查数据目录。")
        sys.exit(1)

    if args.model == "both":
        run_comparison(dataset, val_dataset=val_dataset, **train_kwargs)
    else:
        train_model(args.model, dataset, val_dataset=val_dataset, **train_kwargs)
