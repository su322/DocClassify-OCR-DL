"""
CNN 文档分类模型训练脚本（Baseline）

与 GNN 训练脚本的区别:
- CNN 直接从原始像素图像分类，不需要 OCR 预处理
- 使用标准 torchvision DataLoader + 图像增强
- 训练速度远快于 GNN（不需要逐张跑 OCR）

用法:
    python training/scripts/train_cnn.py --dataset rvl_cdip
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.models.deep_learning.cnn_model import DocumentCNN
from backend.core.config import settings

# ============================================================
# 数据集
# ============================================================


class DocumentImageDataset(Dataset):
    """从文件夹加载文档图像数据集"""

    def __init__(self, root_dir: str, classes: list, transform=None):
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.samples = []

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                ):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 评估
# ============================================================


def evaluate(model, loader, criterion, device):
    """评估模型，返回 loss, accuracy, preds, labels"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# 训练
# ============================================================


def train_cnn(
    data_dir: str,
    classes: list,
    test_dir: str = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    patience: int = 10,
    output_dir: str = "training/output/cnn",
):
    """训练 CNN 模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    train_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"CNN 训练配置 (Baseline)")
    print(f"{'='*60}")
    print(f"  数据目录: {data_dir}")
    print(f"  类别数: {len(classes)}")
    print(f"  批大小: {batch_size}")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")

    # 图像变换
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据
    full_dataset = DocumentImageDataset(data_dir, classes, transform=None)
    if len(full_dataset) == 0:
        print("错误: 没有找到训练数据!")
        return

    # 划分训练集和验证集
    random.shuffle(full_dataset.samples)
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_samples = full_dataset.samples[:-val_size]
    val_samples = full_dataset.samples[-val_size:]

    train_dataset = DocumentImageDataset.__new__(DocumentImageDataset)
    train_dataset.transform = train_transform
    train_dataset.class_to_idx = full_dataset.class_to_idx
    train_dataset.samples = train_samples

    val_dataset = DocumentImageDataset.__new__(DocumentImageDataset)
    val_dataset.transform = eval_transform
    val_dataset.class_to_idx = full_dataset.class_to_idx
    val_dataset.samples = val_samples

    print(f"训练集: {len(train_samples)} 张, 验证集: {len(val_samples)} 张\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 加载测试集（如果存在）
    test_loader = None
    if test_dir and os.path.isdir(test_dir):
        test_dataset = DocumentImageDataset(test_dir, classes, transform=eval_transform)
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            print(f"测试集: {len(test_dataset)} 张\n")

    # 初始化模型
    model = DocumentCNN(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} (ResNet18 预训练)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    # 训练历史
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
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += images.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        avg_val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_preds = val_preds
            best_labels = val_labels
            no_improve_count = 0
            model_path = os.path.join(output_dir, "cnn_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  → 保存最佳模型 (Val Acc: {val_acc:.4f})")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"\n早停触发: 连续 {patience} 轮未改善。")
            break

    train_time = time.time() - train_start_time

    # 加载最佳模型进行最终评估
    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, "cnn_best.pth"),
            map_location=device,
            weights_only=True,
        )
    )

    # 测试集评估
    test_acc = None
    test_preds = None
    test_labels = None
    if test_loader:
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        print(f"\n测试集评估: Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"CNN 训练完成!")
    print(f"  最佳验证集 Acc:  {best_val_acc:.4f}")
    if test_acc is not None:
        print(f"  测试集 Acc:      {test_acc:.4f}")
    print(f"  训练时间:        {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  模型参数量:      {total_params:,}")
    print(f"  训练轮数:        {len(history['train_loss'])}")
    print(f"{'='*60}\n")

    # 保存训练历史
    with open(
        os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 生成可视化（验证集）
    _generate_plots(history, best_preds, best_labels, classes, output_dir, "val")

    # 生成可视化（测试集）
    if test_preds and test_labels:
        _generate_plots(history, test_preds, test_labels, classes, output_dir, "test")

    # 保存完整结果摘要（供对比脚本使用）
    result = {
        "model": "CNN (ResNet18)",
        "val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "total_params": total_params,
        "train_time_seconds": round(train_time, 1),
        "train_epochs": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 4),
    }
    # 计算宏平均 F1
    try:
        from sklearn.metrics import f1_score

        if test_preds and test_labels:
            result["test_macro_f1"] = round(
                f1_score(test_labels, test_preds, average="macro"), 4
            )
        if best_preds and best_labels:
            result["val_macro_f1"] = round(
                f1_score(best_labels, best_preds, average="macro"), 4
            )
    except ImportError:
        pass

    with open(
        os.path.join(output_dir, "result_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def _generate_plots(
    history: dict,
    preds: list,
    labels: list,
    class_names: list,
    output_dir: str,
    split: str = "val",
):
    """生成可视化图表"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        suffix = f"_{split}" if split != "val" else ""

        # Loss/Acc 曲线（只在 val 时生成，因为训练曲线是同一个）
        if split == "val":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(history["train_loss"], label="Train Loss")
            ax1.plot(history["val_loss"], label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("CNN (ResNet18) - Loss Curve")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(history["train_acc"], label="Train Acc")
            ax2.plot(history["val_acc"], label="Val Acc")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("CNN (ResNet18) - Accuracy Curve")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"cnn_loss_acc.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # 混淆矩阵
        if preds and labels:
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(12, 10))
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
            plt.title(f"CNN (ResNet18) - Confusion Matrix ({split})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"cnn_confusion_matrix{suffix}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            report = classification_report(
                labels, preds, target_names=class_names, digits=4
            )
            with open(
                os.path.join(output_dir, f"cnn_classification_report{suffix}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"模型: CNN (ResNet18 Baseline)\n")
                f.write(f"数据集: {split}\n")
                f.write(f"{'='*60}\n\n")
                f.write(report)

        print(f"可视化图表已保存 ({split})")

    except ImportError:
        print("[警告] 缺少 matplotlib/sklearn/seaborn，跳过可视化。")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN 文档分类训练 (Baseline)")
    parser.add_argument("--dataset", type=str, default="rvl_cdip")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None, help="测试集目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="training/output")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.dataset:
        if args.dataset not in settings.DATASETS:
            print(f"错误: 未知数据集 '{args.dataset}'")
            sys.exit(1)
        ds_config = settings.DATASETS[args.dataset]
        data_dir = ds_config["data_dir"]
        classes = ds_config["classes"]
        # 自动推断测试集目录
        if not args.test_dir:
            parent = os.path.dirname(data_dir)
            test_dir = os.path.join(parent, "test")
            if not os.path.isdir(test_dir):
                test_dir = None
        else:
            test_dir = args.test_dir
    elif args.data_dir:
        data_dir = args.data_dir
        classes = settings.DOCUMENT_CLASSES
        test_dir = args.test_dir
    else:
        print("错误: 请指定 --dataset 或 --data-dir")
        sys.exit(1)

    train_cnn(
        data_dir=data_dir,
        classes=classes,
        test_dir=test_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        patience=args.patience,
        output_dir=os.path.join(args.output_dir, "cnn"),
    )
