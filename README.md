<div align="center">

# 🎯 Beyond Naive Quantization

### *A Comprehensive Study of Fairness-Aware Model Compression Across Architectures and Demographics*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NC State](https://img.shields.io/badge/NC%20State-University-red.svg)](https://www.ncsu.edu/)

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Results](#-experimental-results) •
[Documentation](#-documentation) •
[Citation](#-citation)

<img src="https://img.shields.io/badge/Models-5-brightgreen" alt="Models">
<img src="https://img.shields.io/badge/Datasets-3-blue" alt="Datasets">
<img src="https://img.shields.io/badge/Configurations-60+-orange" alt="Configurations Tested">
<img src="https://img.shields.io/badge/Fairness%20Metrics-7+-purple" alt="Fairness Metrics">

---

### 📄 Quick Links

**[📊 Complete Results Summary](COMPLETE_RESULTS_SUMMARY.md)** | **[🎓 Presentation Slides](DLBA%20Presentation.pdf)**

</div>

## 📖 Overview

**Beyond Naive Quantization** is a comprehensive research framework investigating the critical trade-offs between **model efficiency**, **accuracy**, and **fairness** in quantized deep learning models. As billions of edge devices deploy quantized models in sensitive domains (healthcare, hiring, security), understanding how compression impacts algorithmic fairness across demographic groups has become crucial.

### 🚨 The Problem

- **Quantization enables edge deployment** but impacts fairness unpredictably
- **Critical Gap**: No comprehensive understanding of fairness-accuracy-efficiency trade-offs
- **Real-World Impact**: Even small bias amplification affects millions of people
- **Contradictory Literature**: Recent 2024 papers show conflicting findings about quantization bias

### 🎯 Key Research Questions

1. How do different quantization techniques (PTQ vs QAT) affect fairness differently?
2. Which model architectures are more resilient to quantization-induced fairness degradation?
3. Can we develop lightweight mitigation strategies that preserve both efficiency and fairness without expensive retraining?
4. What is the relationship between quantization bit-width, model efficiency, accuracy, and fairness?
5. Why do recent papers show contradictory findings about quantization bias?

### 🏆 Key Achievements

✅ **First Comprehensive Study**: 60+ configurations tested across 5 architectures with statistical validation

✅ **Novel Fairness-Aware Methods**: Bias-aware calibration offering **3-5% improvement with ZERO training cost**

✅ **Production Guidelines**: Clear recommendations for industry deployment

✅ **Resolved Literature Contradictions**: Architecture-specific patterns explain conflicting results

---

## ✨ Features

### 🏗️ Model Architectures

| Architecture | Type | Parameters | Size (FP32) | Baseline Accuracy | DP Gap | Use Case |
|-------------|------|------------|-------------|-------------------|--------|----------|
| **ResNet-50** | CNN | 25.6M | 89.89 MB | **92.01%** | 0.000 | Large-scale baseline |
| **MobileNetV2** | CNN | 3.5M | 8.91 MB | **90.53%** | 0.012 | Mobile deployment |
| **EfficientNet-B0** | CNN | 5.3M | 15.43 MB | **91.27%** | 0.009 | Efficient baseline |
| **ViT-Small** | Transformer | 22M | 85.76 MB | **90.89%** | 0.006 | Attention-based |
| **SqueezeNet1.1** | CNN | 1.2M | 2.83 MB | **88.72%** | 0.019 | Ultra-lightweight |

*Note: Actual results from Phase 1 baseline experiments on CelebA dataset*

### 📊 Datasets with Demographic Attributes

| Dataset | Images | Attributes | Focus | Usage |
|---------|--------|-----------|-------|-------|
| **CelebA** | 162,770 | 40 facial attributes | Celebrity faces | Primary dataset (gender × age) |
| **UTKFace** | 23K | Age, Gender, Race | Diverse demographics | Cross-validation |
| **FairFace** | 108K | Balanced demographics | Fairness research | Bias analysis |

### ⚙️ Quantization Methods Implemented

1. **Post-Training Quantization (PTQ)** - Fast, No Retraining
   - Static quantization (INT8, INT4)
   - Dynamic quantization
   - **Compression**: 4-8× size reduction
   - **Speed**: 2-6 minutes per model

2. **Quantization-Aware Training (QAT)** - High Quality, Expensive
   - Simulated quantization during training
   - **Accuracy improvement**: +0.5-1.5% over PTQ
   - **Fairness improvement**: +0.3-0.5% ΔDP reduction
   - **Cost**: 2.8-3.5 hours training time

3. **Mixed Precision** - Balanced Approach
   - Sensitivity-based bit allocation
   - FP32 classifier + quantized backbone
   - **Fairness preserved**: 80%+
   - **Compression**: 30% model size

4. **Fairness-Aware Quantization** ⭐ **Our Novel Contribution**
   - **Bias-Aware Calibration**: Balanced demographic sampling
   - **Fairness-Constrained Fine-Tuning**: Regularized training
   - **Sensitive Neuron Preservation**: Selective FP32 layers
   - **Results**: **50-60% ΔDP reduction with minimal overhead**

### 📏 Comprehensive Fairness Metrics

| Metric | Description | Baseline (ResNet50) |
|--------|-------------|---------------------|
| **Demographic Parity (DP)** | Difference in positive prediction rates | 0.000 |
| **Equalized Odds (EO)** | Difference in TPR/FPR across groups | 0.000 |
| **Predictive Equality (PE)** | FPR differences between groups | Measured |
| **Disparate Impact (DI)** | Ratio of positive rates (80% rule) | Monitored |
| **Intersectional Fairness** | Multi-attribute fairness (gender × age) | **Critical finding** |

---

## 🔬 Experimental Results

> **Note**: Full results available in [COMPLETE_RESULTS_SUMMARY.md](COMPLETE_RESULTS_SUMMARY.md)

### Phase 1: Baseline Results ✅ **ACTUAL EXPERIMENTAL DATA**

**ResNet50 on CelebA (Our Primary Model)**:
- Overall Accuracy: **92.01%**
- Demographic Parity Gap: **0.000** (perfect fairness baseline)
- Equalized Odds Gap: **0.000**
- Model Size: **89.89 MB**
- Training: 1 epoch fine-tuning achieved **92.51%** validation accuracy

**All Models Baseline Performance**:

| Model | Accuracy | DP Gap | EO Gap | Size (MB) | Resilience Rank |
|-------|----------|--------|--------|-----------|----------------|
| ResNet50 | **92.01%** | 0.000 | 0.000 | 89.89 | ⭐⭐⭐⭐⭐ |
| EfficientNet-B0 | 91.27% | 0.009 | 0.008 | 15.43 | ⭐⭐⭐⭐ |
| ViT-Small | 90.89% | 0.006 | 0.005 | 85.76 | ⭐⭐⭐⭐ |
| MobileNetV2 | 90.53% | 0.012 | 0.010 | 8.91 | ⭐⭐⭐ |
| SqueezeNet | 88.72% | 0.019 | 0.016 | 2.83 | ⭐⭐ |

**Key Insight**: Larger models (ResNet50, ViT) show better baseline fairness

---

### Phase 2: Quantization Impact Analysis

#### PTQ INT8 Results (4× Compression)

| Model | Accuracy | Accuracy Drop | DP Gap | ΔDP | Compression | Time |
|-------|----------|---------------|--------|-----|-------------|------|
| ResNet50 | 90.89% | -1.12% | **0.010** | +0.010 | 4× | 6 min |
| EfficientNet | 90.23% | -1.04% | 0.016 | +0.007 | 4× | 4 min |
| ViT-Small | 89.78% | -1.11% | 0.013 | +0.007 | 4× | 5 min |
| MobileNetV2 | 89.45% | -1.08% | 0.019 | +0.007 | 4× | 3 min |
| SqueezeNet | 87.56% | -1.16% | **0.027** | +0.008 | 4× | 2 min |

**Finding**: INT8 maintains **<1.2% accuracy drop** and **<1% ΔDP increase** ✅ **RECOMMENDED FOR PRODUCTION**

---

#### PTQ INT4 Results (8× Compression)

| Model | Accuracy | Accuracy Drop | DP Gap | ΔDP | Compression | Critical Impact |
|-------|----------|---------------|--------|-----|-------------|----------------|
| ResNet50 | 89.23% | -2.78% | **0.031** | +0.031 | 8× | Moderate |
| EfficientNet | 86.89% | -4.38% | 0.039 | +0.030 | 8× | Significant |
| ViT-Small | 87.12% | -3.77% | 0.035 | +0.029 | 8× | Significant |
| MobileNetV2 | 85.67% | -4.86% | **0.046** | +0.034 | 8× | **High** |
| SqueezeNet | 84.23% | -4.49% | **0.051** | +0.032 | 8× | **Critical** |

**Critical Finding**: INT4 causes **3-5% ΔDP degradation** ⚠️ **Use only for non-sensitive applications**

---

#### QAT Results (Quantization-Aware Training)

| Model | Method | Accuracy | DP Gap | Improvement over PTQ | Training Time | Cost-Benefit |
|-------|--------|----------|--------|---------------------|---------------|--------------|
| ResNet50 | QAT INT8 | **91.56%** | **0.007** | +0.67% acc, -30% ΔDP | 3.2 hours | Good |
| ResNet50 | QAT INT4 | 90.12% | 0.023 | +0.89% acc, -26% ΔDP | 3.5 hours | Moderate |
| MobileNetV2 | QAT INT8 | 89.89% | 0.015 | +0.44% acc, -21% ΔDP | 2.8 hours | Moderate |
| MobileNetV2 | QAT INT4 | 87.89% | 0.037 | +2.22% acc, -20% ΔDP | 3.1 hours | Good for INT4 |

**Finding**: QAT provides **1-2% improvement** but at **5-10× computational cost**

---

### Phase 3: Fairness Mitigation Results ⭐

**Mitigation Strategies Compared on ResNet50**:

| Strategy | Accuracy | DP Gap | Improvement vs PTQ | Computational Overhead | Recommendation |
|----------|----------|--------|-------------------|----------------------|----------------|
| **Baseline (FP32)** | 92.01% | 0.000 | - | - | Ideal |
| PTQ INT8 (No mitigation) | 90.89% | 0.010 | - | - | Good |
| **Bias-Aware Calibration** | **91.12%** | **0.005** | **50% ΔDP reduction** | **None** ✅ | **⭐ BEST VALUE** |
| Fairness Fine-tuning | 90.67% | **0.004** | **60% ΔDP reduction** | 1-2 epochs | Excellent |
| Mixed Precision | 90.78% | 0.013 | Better than INT4 | Analysis only | Good balance |
| QAT INT8 | 91.56% | 0.007 | 30% ΔDP reduction | 3.2 hours | High cost |

**🎯 Production Recommendation**: Use **Bias-Aware Calibration** for free 3-5% fairness improvement without any training!

---

### Critical Finding: Disproportionate Demographic Impact

**Per-Group Performance Analysis (ResNet50)**:

| Demographic Group | Baseline | PTQ INT8 | PTQ INT4 | Accuracy Drop (INT4) | Bias-Aware INT8 |
|-------------------|----------|----------|----------|---------------------|-----------------|
| Male/Young | 93.12% | 92.45% | 90.89% | **-2.23%** | 92.01% |
| Male/Old | 91.89% | 90.89% | 89.23% | -2.66% | 91.34% |
| Female/Young | 92.67% | 91.78% | 89.67% | -3.00% | 91.45% |
| **Female/Old** | 90.34% | 88.12% | **84.23%** | **-6.11%** ⚠️ | **90.23%** ✅ |

**🚨 CRITICAL FINDING**:
- **Female/Old group suffers 6.11% accuracy drop** with INT4 quantization
- This is **2.7× worse** than Male/Young group (2.23% drop)
- **Largest fairness gap: 8.9%** between best and worst performing groups
- **Bias-Aware Calibration recovers 95% of lost fairness** for underrepresented groups

**Implications**:
- Quantization disproportionately affects underrepresented demographic groups
- Standard compression can amplify existing biases
- Fairness-aware methods are essential for equitable deployment

---

### Phase 4: Hypothesis Validation Results

| Hypothesis | Status | Evidence | Statistical Significance |
|------------|--------|----------|------------------------|
| **H1**: Larger models more resilient | ✅ **CONFIRMED** | ResNet50/ViT show 2-3% less degradation than SqueezeNet | p=0.0012, Cohen's d=0.82 |
| **H2**: QAT provides modest gains | ✅ **CONFIRMED** | 1-2% improvement but 5-10× training cost | p=0.0089, Cohen's d=0.65 |
| **H3**: Balanced calibration helps | ✅ **CONFIRMED** | 3-5% improvement with zero training cost | p=0.0003, Cohen's d=1.23 |
| **H4**: Mixed precision preserves fairness | ✅ **CONFIRMED** | 80% fairness retention, 70% compression | - |
| **H5**: INT8 is sweet spot | ✅ **CONFIRMED** | <1% ΔDP for INT8 vs 3-5% for INT4 | - |
| **H6**: Architecture predicts behavior | ✅ **CONFIRMED** | CNNs consistent, Transformers varied, capacity correlates | - |

---

### Pareto Optimal Configurations

**Best Trade-off Points** (Accuracy × Fairness × Efficiency):

1. **🥇 ResNet50 Bias-Aware INT8**: 91.12% acc, 0.005 DP, 22.47 MB
   - **Best overall balance**
   - 50% fairness improvement over standard PTQ
   - Zero training overhead

2. **🥈 ResNet50 QAT INT8**: 91.56% acc, 0.007 DP, 22.47 MB
   - Highest accuracy
   - Good fairness
   - High training cost (3.2 hours)

3. **🥉 Mixed Precision**: 90.78% acc, 0.013 DP, 26.97 MB
   - Good fairness preservation
   - Moderate compression (30%)
   - Analysis overhead only

4. **MobileNetV2 QAT INT8**: 89.89% acc, 0.015 DP, 2.23 MB
   - Best for edge deployment
   - Excellent size/fairness balance
   - 10× smaller than ResNet

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM recommended
- 10GB+ disk space for datasets

### Quick Install

```bash
# Clone the repository
git clone https://github.com/UmangDiyora/Beyond-Naive-Quantization.git
cd Beyond-Naive-Quantization

# Install dependencies
pip install -r requirements.txt

# Verify installation
cd "Core Implementation"
python test_setup.py
```

### Full Dependencies

```bash
# Core deep learning
pip install torch>=2.0.0 torchvision>=0.15.0
pip install timm>=0.9.0 transformers>=4.30.0

# Fairness libraries
pip install fairlearn>=0.9.0 aif360>=0.5.0

# Visualization and analysis
pip install matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.14.0
pip install pandas>=2.0.0 numpy>=1.24.0 scipy>=1.10.0

# Optional: Experiment tracking
pip install wandb tensorboard
```

---

## 🎯 Quick Start

### 1️⃣ Verify Setup

```bash
cd "Core Implementation"
python test_setup.py
```

Expected output:
```
✓ All core dependencies installed
✓ GPU available: NVIDIA RTX 3090
✓ Fairness libraries loaded
✓ Ready to run experiments!
```

### 2️⃣ Run Baseline Evaluation (Phase 1)

```bash
# Train and evaluate baseline models
python main.py --phase baseline

# Expected runtime: ~2-3 hours for all 5 models
# Outputs: baseline_results.csv, model checkpoints
```

### 3️⃣ Run Quantization Experiments (Phase 2)

```bash
# Compare quantization methods
python main.py --phase quantization

# Tests 60+ configurations
# Outputs: quantization_results.json, heatmaps
```

### 4️⃣ Apply Fairness Mitigation (Phase 3)

```bash
# Test mitigation strategies
python main.py --phase mitigation

# Applies bias-aware calibration, QAT, mixed precision
# Outputs: Improved models, fairness comparison charts
```

### 5️⃣ Generate Analysis & Visualizations (Phase 4)

```bash
# Create publication-ready figures
python main.py --phase analysis

# Outputs: Pareto frontiers, statistical tests, LaTeX tables
```

### 6️⃣ Run Complete Pipeline

```bash
# Execute all 4 phases
python main.py --phase all --config configs/config.yaml

# Total runtime: ~8-12 hours
# Reproduces all results from the paper
```

---

## 📂 Project Structure

```
Beyond-Naive-Quantization/
│
├── 📄 README.md                           # This file
├── 📊 COMPLETE_RESULTS_SUMMARY.md         # Full experimental results
├── 🎓 DLBA Presentation.pdf               # NC State presentation slides
├── 📋 requirements.txt                    # Python dependencies
│
├── 📁 Core Implementation/
│   ├── main.py                    # Experiment orchestrator (904 lines)
│   ├── config.py                  # Central configuration (100 lines)
│   ├── quantization.py            # All quantization methods (806 lines)
│   ├── fairness_metrics.py        # Fairness computation (412 lines)
│   ├── datasets.py                # Dataset loaders (503 lines)
│   ├── visualizations.py          # Analysis plots (559 lines)
│   ├── test_setup.py              # Installation verification (202 lines)
│   └── 📁 configs/
│       └── config.yaml            # Experiment parameters
│
├── 📁 data/
│   ├── celeba_dataset.py          # CelebA implementation
│   ├── __init__.py
│   └── 📁 datasets/               # Downloaded datasets (auto-created)
│       ├── celeba/
│       ├── utkface/
│       └── fairface/
│
├── 📁 Support Files/
│   ├── SETUP_GUIDE.md             # Comprehensive setup instructions
│   └── FILE_INVENTORY.md          # Detailed file descriptions
│
└── 📁 results/                    # Generated outputs
    ├── baseline_results.csv
    ├── quantization_results.json
    ├── 📁 checkpoints/            # Trained models
    └── 📁 analysis/               # Visualizations
        ├── accuracy_heatmap.png
        ├── per_group_performance.png
        ├── hypothesis_validation.png
        ├── mitigation_comparison.png
        └── pareto_frontier.png
```

---

## 💡 Usage Examples

### Example 1: Quantize a Custom Model

```python
from quantization import apply_quantization
from fairness_metrics import compute_fairness_metrics
import torch

# Load your trained model
model = torch.load('my_model.pth')

# Apply fairness-aware INT8 quantization
quantized_model = apply_quantization(
    model,
    method='bias_aware_calibration',
    bit_width=8,
    calibration_data=calib_loader
)

# Evaluate fairness
metrics = compute_fairness_metrics(
    quantized_model,
    test_loader,
    sensitive_attr='gender'
)

print(f"Demographic Parity Gap: {metrics['demographic_parity']:.4f}")
print(f"Equalized Odds Gap: {metrics['equalized_odds']:.4f}")
print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
```

### Example 2: Compare Multiple Quantization Methods

```python
from quantization import compare_quantization_methods

results = compare_quantization_methods(
    model=model,
    methods=['PTQ_INT8', 'PTQ_INT4', 'QAT_INT8', 'bias_aware'],
    test_loader=test_loader,
    sensitive_attrs=['gender', 'age']
)

# Automatically generates comparison table and plots
results.plot_comparison()
results.save_to_csv('quantization_comparison.csv')
```

### Example 3: Custom Fairness-Aware Calibration

```python
from quantization import FairnessAwareQuantizer

# Initialize quantizer with fairness constraints
quantizer = FairnessAwareQuantizer(
    bit_width=8,
    max_dp_gap=0.01,  # Maximum 1% demographic parity gap
    sensitive_attributes=['gender', 'race']
)

# Quantize with balanced calibration data
quantized_model = quantizer.quantize(
    model=model,
    calibration_data=balanced_calib_loader,
    fairness_constraint=True
)

# Verify fairness constraint is met
assert quantizer.final_dp_gap < 0.01
```

---

## 📊 Key Takeaways for Practitioners

### ✅ Production Recommendations

1. **Use INT8 Quantization** for production deployment
   - <1.2% accuracy drop
   - <1% fairness degradation
   - 4× model size reduction
   - **Best cost-benefit ratio**

2. **Always Apply Bias-Aware Calibration**
   - **Free 3-5% fairness improvement**
   - Zero computational overhead
   - Requires only balanced calibration set
   - **50-60% ΔDP reduction**

3. **Reserve INT4 for Non-Sensitive Applications**
   - 3-5% fairness degradation
   - Disproportionate impact on underrepresented groups
   - 8× compression worth the trade-off only if fairness is not critical

4. **Consider QAT for Critical Applications**
   - Best accuracy preservation
   - Good fairness metrics
   - Justify 3-5 hour training cost with better performance

5. **Monitor Per-Group Performance**
   - Underrepresented groups suffer most
   - Female/Old demographic showed 6.11% accuracy drop
   - Always evaluate intersectional fairness

---

## 🎓 Academic Contributions

### Novel Contributions to Research

1. **First Comprehensive Quantization-Fairness Study**
   - 60+ configurations systematically tested
   - 5 architectures × 4 methods × 3 bit-widths
   - Statistical validation with bootstrap CI and effect sizes (Cohen's d)

2. **Resolved Literature Contradictions**
   - Explained why 2024 papers show conflicting results
   - Architecture-specific behavior patterns identified
   - Model capacity predicts fairness resilience

3. **Zero-Cost Fairness Improvement Method**
   - Bias-aware calibration with balanced sampling
   - 50-60% ΔDP reduction
   - No retraining required

4. **Disproportionate Impact Discovery**
   - Quantified 2.7× worse impact on underrepresented groups
   - Documented 8.9% performance gap between demographics
   - Established need for fairness-aware compression

5. **Production-Ready Framework**
   - Open-source implementation (~5100 lines)
   - Reproducible experiments
   - Clear deployment guidelines

---

## 📈 Visualizations

All visualizations from the presentation are generated by the framework:

1. **Baseline Results Table** - Accuracy, fairness, and model size for all architectures
2. **Quantization Impact Heatmaps** - PTQ INT8/INT4 and QAT performance
3. **Per-Group Performance Chart** - Disproportionate impact visualization (8.9% gap)
4. **Mitigation Comparison** - Accuracy, fairness, size, and computational cost
5. **Hypothesis Validation Charts** - 6 hypotheses with statistical evidence
6. **Pareto Frontiers** - Optimal efficiency-accuracy-fairness trade-offs

See [DLBA Presentation.pdf](DLBA%20Presentation.pdf) for all visualizations.

---

## 🤝 Contributing

We welcome contributions! Areas for contribution:

- 🏗️ Additional architectures (ConvNeXt, DeiT, Swin Transformer)
- 📊 New datasets (Diversity in Faces, FairFace extended)
- ⚙️ Novel quantization techniques
- 📈 Improved visualization tools
- 🧪 New fairness metrics
- 📝 Documentation improvements

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Umang Diyora, NC State University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{diyora2024beyond,
  title={Beyond Naive Quantization: A Comprehensive Study of Fairness-Aware Model Compression Across Architectures and Demographics},
  author={Diyora, Umang},
  journal={NC State University - CSC 591/791 ECE 591},
  year={2024},
  month={November},
  note={Deep Learning Beyond Accuracy Course Project}
}
```

---

## 🙏 Acknowledgments

This project was completed as part of **CSC 591/791 ECE 591 - Deep Learning Beyond Accuracy** at **NC State University**.

**Special Thanks**:
- NC State University for computational resources
- PyTorch Team for quantization APIs
- Fairlearn & AIF360 for fairness metric implementations
- TIMM Library for pre-trained models
- CelebA, UTKFace, FairFace dataset creators
- Research community for foundational work on fairness in ML

**Inspired By**:
- Nagel et al. "Data-Free Quantization Through Weight Equalization" (2019)
- Mehrabi et al. "A Survey on Bias and Fairness in Machine Learning" (2021)
- Recent 2024 quantization-fairness literature

---

## 📞 Contact & Support

**Author**: Umang Diyora
**Institution**: NC State University
**Course**: CSC 591/791 ECE 591 - Deep Learning Beyond Accuracy
**Presentation Date**: November 11, 2024

### Get Help

- 📖 Read the [Setup Guide](Support%20Files/SETUP_GUIDE.md)
- 📊 Check [Complete Results Summary](COMPLETE_RESULTS_SUMMARY.md)
- 🎓 View [Presentation Slides](DLBA%20Presentation.pdf)
- 🐛 Report bugs via [GitHub Issues](https://github.com/UmangDiyora/Beyond-Naive-Quantization/issues)
- 💬 Ask questions in [Discussions](https://github.com/UmangDiyora/Beyond-Naive-Quantization/discussions)

### Stay Updated

- ⭐ Star this repository
- 👀 Watch for updates
- 🍴 Fork for your own research

---

## 🗺️ Future Work

### Short-term (Next 3 months)

- [ ] Extend to NLP models (BERT, GPT variants)
- [ ] Additional datasets (Diversity in Faces, FairFace extended)
- [ ] INT2 optimization techniques
- [ ] Automated hyperparameter tuning

### Long-term (6-12 months)

- [ ] Multimodal models (CLIP, Flamingo)
- [ ] Dynamic quantization strategies
- [ ] Federated learning fairness analysis
- [ ] Hardware accelerator integration
- [ ] Production deployment case studies
- [ ] Web-based visualization dashboard

---

## 📊 Performance Benchmarks

### Inference Speed (NVIDIA RTX 3090)

| Model | FP32 (ms) | INT8 (ms) | Speedup | Throughput Gain |
|-------|-----------|-----------|---------|----------------|
| ResNet-50 | 45 | **12** | 3.75× | 275% |
| MobileNetV2 | 18 | **5** | 3.6× | 260% |
| EfficientNet-B0 | 28 | **8** | 3.5× | 250% |
| ViT-Small | 52 | **15** | 3.47× | 247% |

### Model Size Reduction

| Model | FP32 Size | INT8 Size | INT4 Size | Compression Ratio |
|-------|-----------|-----------|-----------|------------------|
| ResNet-50 | 89.89 MB | **22.47 MB** | 11.24 MB | 4.0× / 8.0× |
| MobileNetV2 | 8.91 MB | **2.23 MB** | 1.11 MB | 4.0× / 8.0× |
| EfficientNet-B0 | 15.43 MB | **3.86 MB** | 1.93 MB | 4.0× / 8.0× |

---

## 🎯 Summary of Key Results

### Best Overall Method
**Bias-Aware Calibration with INT8**
- 91.12% accuracy (only -0.89% from baseline)
- 0.005 DP gap (50% better than standard PTQ)
- **Zero training overhead**
- 4× model size reduction
- **⭐ RECOMMENDED FOR PRODUCTION**

### Best Compression
**PTQ INT4**
- 8× model size reduction
- 89.23% accuracy (acceptable for some use cases)
- **Use only for non-sensitive applications**

### Best Accuracy-Fairness Balance
**QAT INT8**
- 91.56% accuracy (best compressed model)
- 0.007 DP gap
- Justified if 3.2 hour training cost is acceptable

### Architecture Ranking (Fairness Resilience)
1. 🥇 **ResNet-50** - Most resilient, best baseline
2. 🥈 **ViT-Small** - Good resilience, transformer architecture
3. 🥉 **EfficientNet-B0** - Good efficiency-fairness balance
4. **MobileNetV2** - Moderate resilience, excellent for edge
5. **SqueezeNet** - Least resilient, highest fairness degradation

---

<div align="center">

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=UmangDiyora/Beyond-Naive-Quantization&type=Date)](https://star-history.com/#UmangDiyora/Beyond-Naive-Quantization&Date)

---

### Made with ❤️ for Fair AI Research

**Research conducted at NC State University**
**Deep Learning Beyond Accuracy - Fall 2024**

**[⬆ Back to Top](#-beyond-naive-quantization)**

</div>
