# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intelligent document classification system combining OCR (PaddleOCR PP-StructureV3) and Graph Neural Networks (GCN/GAT/GIN). Processes document images through OCR text+layout extraction, builds graphs with 398-dim node features (384 text embedding + 4 spatial + 10 layout one-hot), and classifies into 16 RVL-CDIP categories.

## Architecture

```
backend/          FastAPI monolith (no async task queue — sync MVP)
├── core/          Settings (Pydantic), SQLAlchemy engine
├── models/
│   ├── database/  SQLAlchemy ORM (DocumentRecord)
│   ├── deep_learning/  PyTorch models: DocumentCNN (ResNet18), DocumentGNN (GCN), DocumentGAT (GAT), DocumentGIN (GIN)
│   └── enums/     DocumentStatus (PENDING/PROCESSING/SUCCESS/FAILED)
├── routers/v1/    POST /api/v1/ocr/process, /api/v1/ocr/classify, /api/v1/classification/predict
├── schemas/       Pydantic request/response models, BaseResponse[T] wrapper
├── services/      OCRService (PPStructureV3 wrapper), ClassificationService (graph builder + GNN inference)
└── crud/          SQLAlchemy CRUD operations
training/          Training infrastructure for CNN/GCN/GAT/GIN on RVL-CDIP
├── scripts/       train_cnn.py, train_gnn.py, prepare_rvl_cdip.py, compare_results.py, test_train.py
└── data/rvl_cdip/ Dataset + caches (OCR base cache + per-strategy graph shards)
tests/             Integration test for PP-StructureV3
frontend/          Unused, not maintained
```

### Data Flow

1. Upload document → OCR (PPStructureV3.predict) → text regions + layout boxes
2. Each region becomes a graph node: SentenceTransformer embedding (384d) + spatial coords (4d) + layout one-hot (10d) = 398d
3. Edges built with chosen strategy:
   - `spatial`: center distance < 15% of document diagonal
   - `reading_order`: connect adjacent regions in reading order (top-to-bottom, left-to-right)
   - `same_row_col`: connect regions sharing the same row or column
   - `hybrid`: union of all three
4. GNN inference → softmax → predicted class + confidence
5. Results saved to SQLite (DocumentRecord)

## Key Commands

### Backend
```bash
python backend/main.py
# API at http://127.0.0.1:8000, Swagger at /docs
```

### Training
```bash
# Full comparison: 3 models × 3 edge strategies (9 runs)
python training/scripts/train_gnn.py --dataset rvl_cdip --batch-size 1024 --in-memory --model all --edge-strategy all

# Single strategy × all models
python training/scripts/train_gnn.py --dataset rvl_cdip --batch-size 1024 --in-memory --model all --edge-strategy spatial

# Single model × all strategies (ablation)
python training/scripts/train_gnn.py --dataset rvl_cdip --batch-size 1024 --in-memory --model gcn --edge-strategy all

# CNN baseline
python training/scripts/train_cnn.py --dataset rvl_cdip

# Data preparation (80/10/10 split)
python training/scripts/prepare_rvl_cdip.py

# Quick validation (2 epochs, 2 samples/class)
python training/scripts/test_train.py

# Compare model results
python training/scripts/compare_results.py
```

### Tests
```bash
python tests/test_pp_structurev3.py
```

### Notable Config
- **Settings**: `backend/core/config.py` — file size limits, database URL, document classes, dataset paths
- **PaddleOCR**: Install paddlepaddle-gpu matching your CUDA version (see requirements.txt comments)
- **HuggingFace mirror**: Set `HF_ENDPOINT=https://hf-mirror.com` on Chinese servers
- **Device**: Auto-selects CUDA > MPS > CPU

### Training Parameters (key defaults)
- CNN: epochs=50, batch_size=128 (or 1024 on GPU), lr=0.001, patience=10
- GNN: epochs=200, batch_size=1024, lr=0.001, patience=20
- GNN models: gcn, gat, gin + both(gcn+gat) + all(gcn+gat+gin)
- GNN edge strategies: spatial, reading_order, same_row_col, hybrid, all
- GNN supports `--in-memory` for servers with large RAM

### Cache Structure (training/data/rvl_cdip/)
```
cache/                  # spatial 建边缓存（旧缓存兼容）
cache_reading_order/    # reading_order 建边缓存
cache_hybrid/           # hybrid 建边缓存
cache_base_train/       # OCR 缓存（所有边策略共享，首次跑一次 OCR 后复用）
cache_base_val/         # 验证集 OCR 缓存
```

OCR 缓存由 `_ensure_ocr_cache()` 维护，各边策略的建边缓存由 `prepare_train_data()` 按策略名写入独立目录。改变边策略无须重新 OCR，只需清对应策略缓存目录。
