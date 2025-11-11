"""
Configuration file for Fairness-Aware Quantization Project
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ProjectConfig:
    """Main configuration class for the project"""
    
    # Project paths
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.abspath(os.path.join(project_root, "..", "data"))
    models_dir: str = os.path.abspath(os.path.join(project_root, "..", "models"))
    results_dir: str = os.path.abspath(os.path.join(project_root, "..", "results"))
    checkpoints_dir: str = os.path.abspath(os.path.join(project_root, "..", "checkpoints"))
    
    # Model architectures
    models: List[str] = [
        "resnet50",
        "mobilenet_v2", 
        "efficientnet_b0",
        "vit_small_patch16_224",
        "squeezenet1_1"
    ]
    
    # Datasets
    datasets: Dict[str, Dict[str, Any]] = {
        "celeba": {
            "path": os.path.join(data_dir, "celeba"),
            "attributes": ["Male", "Young", "Smiling"],
            "protected_attributes": ["Male", "Young"],
            "target_attribute": "Smiling",
            "image_size": 224,
            "num_samples": 202599
        }
    }
    
    # Quantization settings
    quantization: Dict[str, Any] = {
        "methods": ["ptq_static", "ptq_dynamic", "qat", "mixed_precision"],
        "bit_widths": [8, 4, 2],
        "calibration_samples": 1000,
        "qat_epochs": 5,
        "qat_lr": 1e-4
    }
    
    # Fairness metrics
    fairness_metrics: List[str] = [
        "demographic_parity",
        "equalized_odds",
        "predictive_equality",
        "disparate_impact"
    ]
    
    # Training settings
    training: Dict[str, Any] = {
        "batch_size": 32,
        "num_workers": 4,
        "device": "cuda",
        "seed": 42,
        "val_split": 0.2,
        "test_split": 0.2
    }
    
    # Mitigation strategies
    mitigation_strategies: List[str] = [
        "bias_aware_calibration",
        "fairness_constrained_finetuning",
        "sensitive_neuron_preservation",
        "hybrid_precision"
    ]
    
    # Analysis settings
    analysis: Dict[str, Any] = {
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95,
        "statistical_tests": ["paired_t_test", "cohens_d", "bonferroni"],
        "pareto_dimensions": ["model_size", "accuracy", "fairness_gap"]
    }
    
    # Logging and visualization
    logging: Dict[str, Any] = {
        "tensorboard": True,
        "wandb": False,
        "log_interval": 100,
        "save_interval": 1000,
        "visualize_results": True
    }

# Create global config instance
config = ProjectConfig()

# Create necessary directories
os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.models_dir, exist_ok=True)
os.makedirs(config.results_dir, exist_ok=True)
os.makedirs(config.checkpoints_dir, exist_ok=True)
