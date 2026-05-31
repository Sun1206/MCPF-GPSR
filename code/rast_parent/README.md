# RAST: Retrieval-Augmented Spatio-Temporal Framework for Traffic Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2508.16623-b31b1b.svg)](https://arxiv.org/abs/2508.16623)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation of **RAST** (Retrieval-Augmented Spatio-Temporal framework), a universal framework for traffic prediction that integrates retrieval-augmented mechanisms with spatio-temporal modeling.

## Overview

Traffic prediction is a cornerstone of modern intelligent transportation systems. While advanced Spatio-temporal Graph Neural Networks (STGNNs) and pre-trained models have achieved significant progress, two key challenges remain:

1. **Limited contextual capacity** when modeling complex spatio-temporal dependencies
2. **Low predictability** at fine-grained spatio-temporal points due to heterogeneous patterns

Inspired by Retrieval-Augmented Generation (RAG), RAST addresses these challenges through three key components:

- **Decoupled Encoder and Query Generator**: Captures decoupled spatial and temporal features and constructs a fusion query via residual fusion
- **Spatio-temporal Retrieval Store and Retrievers**: Maintains and retrieves vectorized fine-grained patterns
- **Universal Backbone Predictor**: Flexibly accommodates pre-trained STGNNs or simple MLP predictors

## Architecture

```
Input → Decoupled Encoder → Query Generator → Spatio-temporal Retrievers → Backbone Predictor → Output
                                    ↓
                          Retrieval Store (FAISS)
```

## Requirements

### Environment
- Linux Server with CUDA 12.2
- 4x NVIDIA A6000 GPUs (or equivalent)
- Python 3.11
- PyTorch 2.3.1

### Installation

```bash
# Create conda environment
conda create -n RAST python==3.11
conda activate RAST

# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install FAISS for efficient similarity search
pip install faiss-gpu

# Install other dependencies
pip install -r requirements.txt
```

## Datasets

RAST supports multiple real-world traffic datasets:

- **PEMS03, PEMS04, PEMS07, PEMS08**: California PeMS traffic data
- **SD, GBA, GLA**: Large-scale traffic networks

### Dataset Preparation

1. Download datasets from [LargeST](https://github.com/liuxu77/LargeST) or [BasicTS](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/getting_started.md)

2. Organize the downloaded data:
```bash
RAST/
├── datasets/
│   ├── PEMS03/
│   ├── PEMS04/
│   ├── PEMS07/
│   ├── PEMS08/
│   ├── SD/
│   ├── GBA/
│   └── GLA/
```

3. Generate training data:
```bash
cd scripts/data_preparation
bash run.sh
```

## Quick Start
Run all main experiments:
```bash
bash run.sh
```

Or run individual experiments:
```bash
# PEMS datasets
python experiments/train.py -g 0 -c src/scripts/main/RAST_PEMS03.py
python experiments/train.py -g 0 -c src/scripts/main/RAST_PEMS04.py
python experiments/train.py -g 0 -c src/scripts/main/RAST_PEMS07.py
python experiments/train.py -g 0 -c src/scripts/main/RAST_PEMS08.py

# Large-scale datasets
python experiments/train.py -g 0 -c src/scripts/main/RAST_SD.py
python experiments/train.py -g 0 -c src/scripts/main/RAST_GBA.py
python experiments/train.py -g 0 -c src/scripts/main/RAST_GLA.py
```

## Project Structure

```
RAST/
├── basicts/              # Base classes and utilities
│   ├── data/            # Dataset implementations
│   ├── metrics/         # Evaluation metrics (MAE, RMSE, MAPE, etc.)
│   ├── runners/         # Training and evaluation runners
│   ├── scaler/          # Data normalization
│   └── utils/           # Helper functions
├── src/
│   ├── arch/            # RAST architecture
│   │   ├── blocks/      # Building blocks (MLP, RetrievalStore, etc.)
│   │   └── rast_arch.py # Main RAST model
│   ├── scripts/         # Experiment configurations
│   │   ├── main/        # Main experiments
│   │   ├── pretrain/    # Pre-trained backbone experiments
│   │   ├── ablation/    # Ablation studies
│   │   └── hyperparam/  # Hyperparameter studies
│   └── analyze_retrieval_store.py  # Retrieval store analysis tool
├── scripts/
│   ├── data_preparation/  # Dataset preprocessing scripts
│   └── data_visualization/ # Visualization notebooks
├── experiments/
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
└── run.sh               # Batch experiment runner
```

## Analysis Tools

### Analyze Retrieval Store

To visualize and analyze the learned retrieval store:

```bash
cd src
python analyze_retrieval_store.py --file=../database/SD_store_epoch_1.npz
```

This tool helps understand:
- Distribution of retrieved patterns
- Similarity between spatial and temporal retrievals
- Coverage of the retrieval store

## Key Features

- ✅ **Universal Framework**: Works with any STGNN backbone or simple MLP
- ✅ **Efficient Retrieval**: FAISS-based similarity search for scalability
- ✅ **Decoupled Design**: Separate spatial and temporal feature extraction
- ✅ **Fine-grained Patterns**: Retrieves and leverages local spatio-temporal patterns
- ✅ **State-of-the-art Performance**: Superior results on 6 real-world datasets
- ✅ **Computational Efficiency**: Maintains efficiency while improving accuracy

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{ruan2025retrievalaugmentedspatiotemporalframework,
      title={A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction}, 
      author={Weilin Ruan and Xilin Dang and Ziyu Zhou and Sisuo Lyu and Yuxuan Liang},
      year={2025},
      eprint={2508.16623},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.16623}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.