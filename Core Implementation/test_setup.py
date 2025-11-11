#!/usr/bin/env python
"""
Quick test script to verify the fairness-aware quantization setup
Run this to make sure all modules are working correctly
"""

import sys
import os

# Add project root and current module directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing config.py: {e}")
        return False
    
    try:
        from data import celeba_dataset
        print("✓ data.celeba_dataset imported successfully")
    except Exception as e:
        print(f"✗ Error importing data.celeba_dataset: {e}")
        return False
    
    try:
        import models
        print("✓ models.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing models.py: {e}")
        return False
    
    try:
        import quantization
        print("✓ quantization.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing quantization.py: {e}")
        return False
    
    try:
        import fairness_metrics
        print("✓ fairness_metrics.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing fairness_metrics.py: {e}")
        return False
    
    try:
        import visualizations
        print("✓ visualizations.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing visualizations.py: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test fairness metrics
        from fairness_metrics import FairnessMetrics
        import numpy as np
        
        fm = FairnessMetrics()
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Calculate metrics
        metrics = fm.calculate_all_metrics(y_true, y_pred, sensitive)
        
        print("✓ Fairness metrics calculation working")
        print(f"  - Accuracy: {metrics['accuracy']:.3f}")
        print(f"  - Demographic Parity: {metrics['demographic_parity']:.3f}")
        
    except Exception as e:
        print(f"✗ Error testing fairness metrics: {e}")
        return False
    
    try:
        # Test model loading (without actually loading weights)
        import torch
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.fc = nn.Linear(16 * 224 * 224, 2)
            
            def forward(self, x):
                x = self.conv1(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel()
        print("✓ Model creation working")
        
        # Test quantization configuration
        from quantization import PostTrainingQuantization
        quantizer = PostTrainingQuantization(bit_width=8)
        print("✓ Quantization module working")
        
    except Exception as e:
        print(f"✗ Error testing models/quantization: {e}")
        return False
    
    return True


def test_visualization():
    """Test visualization capabilities"""
    print("\nTesting visualization...")
    
    try:
        from visualizations import FairnessVisualizer
        import numpy as np
        
        visualizer = FairnessVisualizer()
        
        # Create dummy results
        dummy_results = {
            'test_model_baseline': {
                'metrics': {
                    'model_name': 'test_model',
                    'quantization_method': 'baseline',
                    'bit_width': 32,
                    'accuracy': 0.95,
                    'demographic_parity': 0.05,
                    'equalized_odds': 0.04
                }
            },
            'test_model_quantized': {
                'metrics': {
                    'model_name': 'test_model',
                    'quantization_method': 'ptq',
                    'bit_width': 8,
                    'accuracy': 0.92,
                    'demographic_parity': 0.08,
                    'equalized_odds': 0.07
                }
            }
        }
        
        print("✓ Visualization module working")
        print("  (Note: Actual plots require matplotlib/plotly backends)")
        
    except Exception as e:
        print(f"✗ Error testing visualization: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Fairness-Aware Quantization Setup Test")
    print("="*50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test visualization
    if not test_visualization():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ ALL TESTS PASSED - Setup is working correctly!")
        print("\nYou can now run the main experiment with:")
        print("  python main.py --phase all --debug")
    else:
        print("✗ Some tests failed - please check the errors above")
        print("\nCommon issues:")
        print("  - Missing dependencies: run 'pip install -r requirements.txt'")
        print("  - Import errors: ensure all files are in the same directory")
    print("="*50)


if __name__ == '__main__':
    main()
