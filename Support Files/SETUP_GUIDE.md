# Fairness-Aware Model Compression Project
## Complete Implementation Guide

This repository contains the complete implementation for the paper:
**"Beyond Naive Quantization: A Comprehensive Study of Fairness-Aware Model Compression Across Architectures and Demographics"**

## 📁 Project Structure

```
fairness_quantization/
│
├── config.py                 # Main configuration file
├── datasets.py              # Dataset loaders (CelebA, UTKFace, FairFace)
├── models.py                # Model architectures and loading
├── quantization.py          # Quantization methods (PTQ, QAT, Mixed Precision)
├── fairness_metrics.py      # Fairness metrics implementation
├── main.py                  # Main experiment runner
├── visualizations.py        # Result visualization tools
├── requirements.txt         # Python dependencies
│
├── configs/                 # Configuration files
│   └── config.yaml         # YAML config (alternative to config.py)
│
├── data/                   # Dataset utilities
│   ├── __init__.py
│   └── celeba_dataset.py
│
├── models/                 # Model utilities
│   └── quantization_utils.py
│
├── metrics/               # Metrics utilities
│   └── fairness_metrics.py
│
└── analysis/             # Analysis tools
    └── visualizations.py
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this directory
cd fairness_quantization

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the datasets and organize them as follows:

```
data/
├── celeba/
│   ├── img_align_celeba/     # Face images
│   ├── list_attr_celeba.txt  # Attributes file
│   └── list_eval_partition.txt  # Train/val/test splits
│
├── utkface/
│   └── *.jpg  # Images with format: [age]_[gender]_[race]_[date].jpg
│
└── fairface/
    ├── train/
    ├── val/
    ├── fairface_label_train.csv
    └── fairface_label_val.csv
```

### 3. Running Experiments

#### Run All Phases (Recommended for first run)
```bash
python main.py --phase all
```

#### Run Individual Phases
```bash
# Phase 1: Baseline evaluation
python main.py --phase 1

# Phase 2: Quantization comparison
python main.py --phase 2

# Phase 3: Fairness mitigation
python main.py --phase 3

# Phase 4: Analysis and visualization
python main.py --phase 4
```

#### Debug Mode (Faster, smaller datasets)
```bash
python main.py --phase all --debug
```

## 📊 Key Experiments

### Phase 1: Infrastructure & Baseline (Weeks 1-2)
- Sets up infrastructure
- Loads and prepares datasets
- Fine-tunes baseline models
- Evaluates baseline fairness metrics

### Phase 2: Quantization Comparison (Weeks 3-4)
- **PTQ (Post-Training Quantization)**: INT8, INT4
- **QAT (Quantization-Aware Training)**: With simulated quantization
- **Mixed Precision**: Different bit-widths per layer
- Tests 5 models × 4 methods × 3 bit-widths = 60 configurations

### Phase 3: Fairness Mitigation (Weeks 5-6)
- **Bias-aware calibration**: Balanced demographic sampling
- **Fairness-constrained fine-tuning**: With DP regularization
- **Sensitive neuron preservation**: Higher precision for bias-sensitive layers
- **Hybrid approaches**: FP32 classifier + INT4 backbone

### Phase 4: Analysis (Weeks 7-9)
- Architecture resilience comparison
- Pareto frontier analysis (3D: efficiency-accuracy-fairness)
- Statistical significance testing
- Comprehensive visualizations

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Key settings
config = ProjectConfig(
    models=['resnet50', 'mobilenet_v2', 'efficientnet_b0', 'vit_small', 'squeezenet1_1'],
    datasets={
        'celeba': {...},
        'utkface': {...},
        'fairface': {...}
    },
    quantization={
        'methods': ['ptq_static', 'ptq_dynamic', 'qat', 'mixed_precision'],
        'bit_widths': [8, 4, 2],
        'calibration_samples': 1000,
        'qat_epochs': 5
    },
    fairness_metrics=['demographic_parity', 'equalized_odds', 'predictive_equality'],
    ...
)
```

## 📈 Expected Results

### Hypothesis Testing
- **H1**: Larger models (ResNet-50, ViT) preserve fairness better than lightweight models
- **H2**: QAT provides modest gains (1-2% ΔDP) at high computational cost
- **H3**: Balanced calibration improves fairness by 3-5% with no training
- **H4**: Mixed-precision with FP32 classifier preserves 80%+ fairness
- **H5**: INT8 is the sweet spot (<1% ΔDP), INT4 causes 3-5% degradation
- **H6**: Architecture predicts bias behavior (CNNs vs Transformers)

## 📉 Visualization

The project generates comprehensive visualizations:

```python
from visualizations import FairnessVisualizer

visualizer = FairnessVisualizer()

# Generate all visualizations
visualizer.create_comprehensive_report(
    all_results, 
    output_dir='./visualizations'
)
```

Outputs include:
- Accuracy heatmaps across models/quantization levels
- 3D Pareto frontiers (efficiency-accuracy-fairness)
- Per-demographic group performance
- Layer sensitivity analysis
- LaTeX tables for papers

## 🎯 Custom Usage Examples

### Example 1: Quantize a Single Model
```python
from models import load_and_prepare_models
from quantization import PostTrainingQuantization
from datasets import get_fairness_dataloader

# Load model
models = load_and_prepare_models(['resnet50'], num_classes=2)
model = models['resnet50']

# Get calibration data
cal_loader = get_fairness_dataloader('celeba', './data/celeba', split='train', subsample=1000)

# Quantize
quantizer = PostTrainingQuantization(bit_width=8)
quantized_model = quantizer.quantize(model, cal_loader)
```

### Example 2: Evaluate Fairness
```python
from fairness_metrics import FairnessMetrics

metrics_calc = FairnessMetrics()

# Calculate all fairness metrics
results = metrics_calc.calculate_all_metrics(
    y_true, y_pred, sensitive_features,
    return_details=True
)

print(f"Demographic Parity Gap: {results['demographic_parity']:.3f}")
print(f"Equalized Odds Gap: {results['equalized_odds']:.3f}")
```

### Example 3: Fairness-Aware Quantization
```python
from quantization import FairnessAwareQuantization

# Create fairness-aware quantizer
fa_quantizer = FairnessAwareQuantization(
    base_method='ptq',
    fairness_constraint='demographic_parity',
    fairness_weight=0.1
)

# Apply bias-aware calibration
quantized_model = fa_quantizer.bias_aware_calibration(
    model, calibration_loader, sensitive_groups=[0, 1, 2, 3]
)
```

## 📝 Key Files Explained

### `config.py`
Central configuration for all experiments. Modify this to change:
- Model architectures to test
- Datasets and attributes
- Quantization methods and bit-widths
- Fairness metrics
- Training hyperparameters

### `datasets.py`
Implements dataset loaders for:
- **CelebA**: 200K celebrity faces with 40 attributes
- **UTKFace**: 23K faces with age, gender, race labels
- **FairFace**: 108K balanced faces across demographics

### `models.py`
- Loads pre-trained models (ResNet, MobileNet, EfficientNet, ViT, SqueezeNet)
- Provides model profiling (size, FLOPs, inference time)
- Handles fine-tuning on fairness datasets

### `quantization.py`
Implements quantization methods:
- **PTQ**: Post-training with calibration
- **QAT**: Simulated quantization during training
- **Mixed Precision**: Layer-wise bit allocation
- **Fairness-Aware**: Bias-aware calibration, fairness-constrained training

### `fairness_metrics.py`
Comprehensive fairness metrics:
- Demographic Parity (DP)
- Equalized Odds (EO)
- Predictive Equality (PE)
- Disparate Impact (DI)
- Intersectional fairness
- Statistical significance testing

### `main.py`
Main experiment orchestrator that:
- Runs all experimental phases
- Saves results and checkpoints
- Generates analysis reports
- Handles experiment configuration

### `visualizations.py`
Creates publication-ready figures:
- Heatmaps, 3D scatter plots
- Fairness comparison charts
- Per-group performance analysis
- LaTeX tables for papers

## 🐛 Troubleshooting

### CUDA/GPU Issues
```python
# Force CPU usage if GPU unavailable
config.training['device'] = 'cpu'
```

### Memory Issues
```python
# Reduce batch size
config.training['batch_size'] = 16

# Use data subsampling
train_loader = get_fairness_dataloader(..., subsample=1000)
```

### Dataset Loading
Ensure datasets are properly organized and paths are correct in `config.py`:
```python
config.data_dir = '/path/to/your/data'
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{fairness_quantization_2024,
  title={Beyond Naive Quantization: A Comprehensive Study of 
         Fairness-Aware Model Compression Across Architectures 
         and Demographics},
  author={Your Name},
  journal={CSC 591/791 ECE 591 Deep Learning Beyond Accuracy},
  year={2024}
}
```

## 🔑 Key Contributions

1. **Empirical Evidence**: First comprehensive quantization-fairness comparison across architectures
2. **Practical Guidelines**: Actionable recommendations for fair compressed model deployment
3. **Theoretical Insights**: Understanding of architecture-specific resilience to quantization bias
4. **Lightweight Mitigations**: Cost-effective strategies without expensive retraining
5. **Literature Resolution**: Reconciliation of contradictory 2024 findings
6. **Open Source**: Complete reproducible implementation

## 💡 Tips for Best Results

1. **Start with debug mode** to verify setup works
2. **Use balanced calibration sets** for better fairness
3. **Monitor per-group metrics**, not just overall accuracy
4. **Test multiple random seeds** for statistical significance
5. **Save intermediate checkpoints** for long experiments

## 📧 Support

For questions or issues with the code, please check:
1. This README file
2. Comments in the source code
3. The original project abstract (included in repo)

---

**Note**: This implementation is designed for research and educational purposes. Always validate results on your specific use case before deployment.
