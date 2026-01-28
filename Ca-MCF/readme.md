# Ca-MCF: Category-aware Multi-label Causal Feature Selection

## Introduction

**Ca-MCF** is a causal feature selection algorithm specifically designed for **Multi-label Learning**. The core innovation lies in shifting from traditional **label-level** feature selection to a more granular **category-level** analysis. By modeling dependencies between specific label categories and utilizing **Single-Category Specific Mutual Information (SCSMI)** and **Dual-Category Specific Mutual Information (DCSMI)**, the algorithm discovers local causal structures to accurately identify feature subsets with explanatory power for specific label values.

## Project Structure

```text
.
├── CaMCF.m                # Core algorithm implementation
├── example_run.m          # Main execution script (data loading & evaluation)
├── classifiers/           # Classification models
│   └── ML_KNN.m           # Standard ML-kNN implementation
├── common/                # Core mathematical tools
│   ├── DCSMI.m            # Dual-Category Specific Mutual Information
│   └── SCSMI.m            # Single-Category Specific Mutual Information
├── data/                  # Dataset directory (e.g., Flags, CHD_49, Yeast, etc.)
└── evaluation/            # Performance metric functions
    ├── calculate_hamming_loss.m
    ├── calculate_subset_accuracy.m
    └── ... (Macro-F1, Micro-F1, Ranking Metrics)
```

## Quick Start

### 1. Prerequisites

- **MATLAB R2021a** or later.
- Ensure the `common`, `classifiers`, and `evaluation` folders are added to your MATLAB path.

### 2. Running the Example

Execute the following in the MATLAB command window:

```matlab
example_run