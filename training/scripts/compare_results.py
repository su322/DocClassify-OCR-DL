"""
模型对比结果汇总脚本

读取三个模型的 result_summary.json，生成论文用的对比表格。

用法:
    python training/scripts/compare_results.py
    python training/scripts/compare_results.py --output-dir training/output
"""

import os
import sys
import json
import argparse

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DEFAULT_OUTPUT_DIR = "training/output"


def load_result(model_dir: str) -> dict:
    """加载单个模型的结果摘要"""
    path = os.path.join(model_dir, "result_summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_table(results: dict) -> str:
    """生成 Markdown 格式的对比表格"""
    lines = []
    lines.append(
        "| 模型 | 验证集准确率 | 验证集 Macro-F1 | 测试集准确率 | 测试集 Macro-F1 | 参数量 | 训练时间 | 训练轮数 |"
    )
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for name, r in results.items():
        val_acc = (
            f"{r.get('val_accuracy', 'N/A'):.4f}" if r.get("val_accuracy") else "N/A"
        )
        val_f1 = (
            f"{r.get('val_macro_f1', 'N/A'):.4f}" if r.get("val_macro_f1") else "N/A"
        )
        test_acc = (
            f"{r.get('test_accuracy', 'N/A'):.4f}" if r.get("test_accuracy") else "N/A"
        )
        test_f1 = (
            f"{r.get('test_macro_f1', 'N/A'):.4f}" if r.get("test_macro_f1") else "N/A"
        )
        params = f"{r.get('total_params', 0):,}" if r.get("total_params") else "N/A"
        train_time = (
            f"{r.get('train_time_seconds', 0):.1f}s"
            if r.get("train_time_seconds")
            else "N/A"
        )
        epochs = str(r.get("train_epochs", "N/A")) if r.get("train_epochs") else "N/A"
        lines.append(
            f"| {name} | {val_acc} | {val_f1} | {test_acc} | {test_f1} | {params} | {train_time} | {epochs} |"
        )

    return "\n".join(lines)


def format_latex_table(results: dict) -> str:
    """生成 LaTeX 格式的对比表格（论文直接用）"""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{不同模型在 RVL-CDIP 数据集上的性能对比}")
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"模型 & 验证集准确率 & 验证集F1 & 测试集准确率 & 测试集F1 & 参数量 & 训练时间 \\"
    )
    lines.append(r"\midrule")

    for name, r in results.items():
        val_acc = f"{r.get('val_accuracy', 0):.4f}" if r.get("val_accuracy") else "-"
        val_f1 = f"{r.get('val_macro_f1', 0):.4f}" if r.get("val_macro_f1") else "-"
        test_acc = f"{r.get('test_accuracy', 0):.4f}" if r.get("test_accuracy") else "-"
        test_f1 = f"{r.get('test_macro_f1', 0):.4f}" if r.get("test_macro_f1") else "-"
        params = (
            f"{r.get('total_params', 0)/1e6:.2f}M" if r.get("total_params") else "-"
        )
        train_time = (
            f"{r.get('train_time_seconds', 0)/60:.1f}min"
            if r.get("train_time_seconds")
            else "-"
        )
        lines.append(
            f"{name} & {val_acc} & {val_f1} & {test_acc} & {test_f1} & {params} & {train_time} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="模型对比结果汇总")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir

    # 加载所有模型的结果
    model_dirs = {
        "CNN (ResNet18)": os.path.join(output_dir, "cnn"),
        "GCN": os.path.join(output_dir, "gcn"),
        "GAT": os.path.join(output_dir, "gat"),
    }

    results = {}
    for name, dir_path in model_dirs.items():
        r = load_result(dir_path)
        if r:
            results[name] = r
            print(f"  ✅ 已加载: {name}")
        else:
            print(f"  ⚠️  未找到: {name} ({dir_path}/result_summary.json)")

    if not results:
        print("\n错误: 没有找到任何模型结果。请先运行训练脚本。")
        return

    print(f"\n{'='*60}")
    print(f"  模型对比结果")
    print(f"{'='*60}\n")

    # 打印表格
    md_table = format_table(results)
    print(md_table)

    # 保存 Markdown 表格
    md_path = os.path.join(output_dir, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 模型对比结果\n\n")
        f.write(md_table)
        f.write("\n")

    # 保存 LaTeX 表格
    latex_table = format_latex_table(results)
    latex_path = os.path.join(output_dir, "comparison_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
        f.write("\n")

    # 保存 JSON 汇总
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n文件已保存:")
    print(f"  Markdown 表格: {md_path}")
    print(f"  LaTeX 表格:    {latex_path}")
    print(f"  JSON 汇总:     {json_path}")


if __name__ == "__main__":
    main()
