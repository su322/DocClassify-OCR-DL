# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intelligent document classification system combining OCR (PaddleOCR PP-StructureV3) and Graph Neural Networks (GCN/GAT). Processes document images through OCR text+layout extraction, builds graphs with 398-dim node features (384 text embedding + 4 spatial + 10 layout one-hot), and classifies into 16 RVL-CDIP categories.

## Architecture

```
backend/          FastAPI monolith (no async task queue — sync MVP)
├── core/          Settings (Pydantic), SQLAlchemy engine
├── models/
│   ├── database/  SQLAlchemy ORM (DocumentRecord)
│   ├── deep_learning/  PyTorch models: DocumentCNN (ResNet18), DocumentGNN (GCN), DocumentGAT (GAT)
│   └── enums/     DocumentStatus (PENDING/PROCESSING/SUCCESS/FAILED)
├── routers/v1/    POST /api/v1/ocr/process, /api/v1/ocr/classify, /api/v1/classification/predict
├── schemas/       Pydantic request/response models, BaseResponse[T] wrapper
├── services/      OCRService (PPStructureV3 wrapper), ClassificationService (graph builder + GNN inference)
└── crud/          SQLAlchemy CRUD operations
training/          Training infrastructure for CNN/GCN/GAT on RVL-CDIP
├── scripts/       train_cnn.py, train_gnn.py, prepare_rvl_cdip.py, compare_results.py, test_train.py
└── data/rvl_cdip/ Dataset + graph cache shards (.pt files, 1000 samples each)
tests/             Integration test for PP-StructureV3
frontend/          Unused, not maintained
```

### Data Flow

1. Upload document → OCR (PPStructureV3.predict) → text regions + layout boxes
2. Each region becomes a graph node: SentenceTransformer embedding (384d) + spatial coords (4d) + layout one-hot (10d) = 398d
3. Edges connect nodes whose center distance < 15% of document diagonal
4. GNN inference → softmax → predicted class + confidence
5. Results saved to SQLite (DocumentRecord)

## Key Commands

### Backend
```bash
# Start the API server (hot reload enabled)
python backend/main.py
# API at http://127.0.0.1:8000
# Swagger: http://127.0.0.1:8000/docs
```

### Training
```bash
# CNN baseline (ResNet18)
python training/scripts/train_cnn.py --dataset rvl_cdip

# GNN training (GCN, GAT, or both)
python training/scripts/train_gnn.py --dataset rvl_cdip --model both

# Data preparation (80/10/10 split)
python training/scripts/prepare_rvl_cdip.py

# Quick validation (2 epochs, 2 samples/class)
python training/scripts/test_train.py

# Compare model results
python training/scripts/compare_results.py
```

### Tests
```bash
# PP-StructureV3 integration test
python tests/test_pp_structurev3.py
```

### Notable Config
- **Settings**: `backend/core/config.py` — file size limits, database URL, document classes, dataset paths
- **PaddleOCR**: Install paddlepaddle-gpu matching your CUDA version (see requirements.txt comments)
- **HuggingFace mirror**: Set `HF_ENDPOINT=https://hf-mirror.com` on Chinese servers
- **Device**: Auto-selects CUDA > MPS > CPU

### Training Parameters (key defaults)
- CNN: epochs=50, batch_size=128, lr=0.001, patience=10
- GNN: epochs=200, batch_size=128, lr=0.001, patience=20
- GNN supports `--in-memory` for servers with large RAM, and graph shard caching by default (~500MB memory)

### Graph Cache
Preprocessed graph shards stored in `training/data/rvl_cdip/cache/`. The `GraphDataset` uses LRU shard caching — delete cache to force re-preprocessing.
