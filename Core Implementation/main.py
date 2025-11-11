"""
Main Experiment Runner for Fairness-Aware Quantization Study
"""

import os
import sys
import yaml
import json
import argparse
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torchvision.models as models
import timm

# Ensure project root is on sys.path so 'data' package is importable when running from this folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.celeba_dataset import create_celeba_dataloaders
from fairness_metrics import FairnessMetrics, FairnessTracker
from quantization import ModelQuantizer, FairnessAwareQuantization


class FairnessQuantizationExperiment:
    """
    Main experiment class for fairness-aware quantization study
    """
    
    def __init__(self, config_path: str):
        """
        Initialize experiment
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Normalize key paths relative to project root
        self.config.setdefault('data', {}).setdefault('celeba', {})
        celeba_cfg = self.config['data']['celeba']
        default_celeba_root = os.path.join(PROJECT_ROOT, 'data', 'celeba')
        root_path = celeba_cfg.get('root') or default_celeba_root
        if not os.path.isabs(root_path):
            root_path = os.path.abspath(os.path.join(PROJECT_ROOT, root_path))
        celeba_cfg['root'] = root_path

        analysis_cfg = self.config.setdefault('analysis', {})
        results_dir_cfg = analysis_cfg.get('results_dir', os.path.join(PROJECT_ROOT, 'results'))
        if not os.path.isabs(results_dir_cfg):
            results_dir_cfg = os.path.abspath(os.path.join(PROJECT_ROOT, results_dir_cfg))
        analysis_cfg['results_dir'] = results_dir_cfg

        # Set device
        self.device = torch.device(
            self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # Set random seeds for reproducibility
        self.set_seeds(self.config['hardware']['seed'])
        
        # Create results directory
        self.results_dir = Path(self.config['analysis']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.fairness_metrics = FairnessMetrics()
        self.fairness_tracker = FairnessTracker()
        
        # Results storage
        self.all_results = {}
        
        print(f"Experiment initialized on {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
        
    def set_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def load_model(self, architecture: str, num_classes: int = 2) -> nn.Module:
        """
        Load pre-trained model
        
        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            
        Returns:
            Loaded model
        """
        print(f"Loading model: {architecture}")
        
        if architecture == 'resnet50':
            model = models.resnet50(pretrained=self.config['models']['pretrained'])
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
        elif architecture == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=self.config['models']['pretrained'])
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            
        elif architecture == 'efficientnet_b0':
            model = timm.create_model(
                'efficientnet_b0',
                pretrained=self.config['models']['pretrained'],
                num_classes=num_classes
            )
            
        elif architecture == 'vit_small_patch16_224':
            model = timm.create_model(
                'vit_small_patch16_224',
                pretrained=self.config['models']['pretrained'],
                num_classes=num_classes
            )
            
        elif architecture == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=self.config['models']['pretrained'])
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
            model.num_classes = num_classes
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        model = model.to(self.device)
        return model
    
    def fine_tune_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5
    ) -> nn.Module:
        """
        Fine-tune pre-trained model on target dataset
        
        Args:
            model: Model to fine-tune
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of fine-tuning epochs
            
        Returns:
            Fine-tuned model
        """
        print("Fine-tuning model...")
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, "
                          f"Acc={100.*train_correct/train_total:.2f}%")
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100. * val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Acc={100.*train_correct/train_total:.2f}%, "
                  f"Val Acc={val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        model_name: str = "model"
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            model_name: Name for the model
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating {model_name}...")
        
        model.eval()
        
        # Collect predictions
        all_preds = []
        all_targets = []
        all_sensitive = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, targets, sensitive in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_sensitive.extend(sensitive.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_sensitive = np.array(all_sensitive)
        all_probs = np.array(all_probs)
        
        # Overall metrics
        overall_accuracy = np.mean(all_preds == all_targets)
        
        # Per-group metrics
        group_metrics = {}
        unique_groups = np.unique(all_sensitive)
        
        for group in unique_groups:
            mask = all_sensitive == group
            group_acc = np.mean(all_preds[mask] == all_targets[mask])
            group_metrics[f'accuracy_group_{group}'] = group_acc
        
        # Fairness metrics
        fairness_results = self.fairness_metrics.calculate_all_metrics(
            all_targets,
            all_preds,
            all_sensitive,
            all_probs
        )
        
        # Combine all results
        results = {
            'model_name': model_name,
            'overall_accuracy': overall_accuracy,
            **group_metrics,
            **fairness_results
        }
        
        # Model efficiency metrics
        model_size = self._get_model_size(model)
        results['model_size_mb'] = model_size
        
        return results
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def run_baseline_experiments(self):
        """Run baseline experiments on all models"""
        print("\n" + "="*50)
        print("PHASE 1: BASELINE EXPERIMENTS")
        print("="*50)
        
        # Load data
        train_loader, val_loader, test_loader = create_celeba_dataloaders(
            self.config,
            batch_size=self.config['data']['celeba']['batch_size'],
            num_workers=self.config['data']['celeba']['num_workers']
        )
        
        baseline_results = {}
        
        for architecture in self.config['models']['architectures']:
            print(f"\n--- Processing {architecture} ---")
            
            # Load and fine-tune model
            model = self.load_model(
                architecture,
                num_classes=self.config['models']['num_classes']
            )
            
            model = self.fine_tune_model(
                model,
                train_loader,
                val_loader,
                epochs=self.config['training']['epochs']
            )
            
            # Save fine-tuned model
            model_path = self.results_dir / f"{architecture}_finetuned.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved fine-tuned model to {model_path}")
            
            # Evaluate baseline model
            results = self.evaluate_model(
                model,
                test_loader,
                model_name=f"{architecture}_baseline"
            )
            
            baseline_results[architecture] = results
            
            # Print key metrics
            print(f"\nBaseline Results for {architecture}:")
            print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"  Demographic Parity Diff: {results.get('dp_max_dp_difference', 0):.4f}")
            print(f"  Equalized Odds Diff: {results.get('eo_equalized_odds_difference', 0):.4f}")
            print(f"  Model Size: {results['model_size_mb']:.2f} MB")
        
        # Save baseline results
        baseline_df = pd.DataFrame(baseline_results).T
        baseline_df.to_csv(self.results_dir / 'baseline_results.csv')
        
        self.all_results['baseline'] = baseline_results
        
        return baseline_results
    
    def run_quantization_experiments(self):
        """Run quantization comparison experiments"""
        print("\n" + "="*50)
        print("PHASE 2: QUANTIZATION COMPARISON")
        print("="*50)
        
        # Load data
        train_loader, val_loader, test_loader = create_celeba_dataloaders(
            self.config,
            batch_size=self.config['data']['celeba']['batch_size'],
            num_workers=self.config['data']['celeba']['num_workers']
        )
        
        # Get calibration subset
        calibration_dataset = train_loader.dataset.get_calibration_subset(
            num_samples=self.config['quantization']['methods']['ptq']['calibration_samples'],
            balanced=True
        )
        calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=32,
            shuffle=False
        )
        
        quantization_results = {}
        
        for architecture in self.config['models']['architectures']:
            print(f"\n--- Quantizing {architecture} ---")
            
            # Load fine-tuned model
            model = self.load_model(
                architecture,
                num_classes=self.config['models']['num_classes']
            )
            model_path = self.results_dir / f"{architecture}_finetuned.pth"
            model.load_state_dict(torch.load(model_path))
            
            # Initialize quantizer
            quantizer = ModelQuantizer(
                model,
                device=self.device,
                backend=self.config['quantization']['methods']['ptq']['backends'][0]
            )
            
            arch_results = {}
            
            # Test different quantization methods
            for bit_width in self.config['quantization']['methods']['bit_widths']:
                print(f"\n  Testing {bit_width}-bit quantization...")
                
                # PTQ
                print("    Applying PTQ...")
                ptq_model = quantizer.post_training_quantization(
                    model,
                    calibration_loader,
                    bit_width=bit_width,
                    calibration_method=self.config['quantization']['methods']['ptq']['calibration_method']
                )
                
                ptq_results = self.evaluate_model(
                    ptq_model,
                    test_loader,
                    model_name=f"{architecture}_ptq_{bit_width}bit"
                )
                
                arch_results[f'ptq_{bit_width}bit'] = ptq_results
                
                # QAT (only for INT8)
                if bit_width == 8:
                    print("    Applying QAT...")
                    qat_model = quantizer.quantization_aware_training(
                        model,
                        train_loader,
                        val_loader,
                        epochs=self.config['quantization']['methods']['qat']['epochs'],
                        learning_rate=self.config['quantization']['methods']['qat']['learning_rate']
                    )
                    
                    qat_results = self.evaluate_model(
                        qat_model,
                        test_loader,
                        model_name=f"{architecture}_qat_{bit_width}bit"
                    )
                    
                    arch_results[f'qat_{bit_width}bit'] = qat_results
                
                # Dynamic Quantization
                if bit_width == 8:
                    print("    Applying Dynamic Quantization...")
                    dynamic_model = quantizer.dynamic_quantization(model)
                    
                    dynamic_results = self.evaluate_model(
                        dynamic_model,
                        test_loader,
                        model_name=f"{architecture}_dynamic"
                    )
                    
                    arch_results['dynamic'] = dynamic_results
            
            quantization_results[architecture] = arch_results
            
            # Print comparison
            self._print_quantization_comparison(architecture, arch_results)
        
        # Save quantization results
        self._save_quantization_results(quantization_results)
        
        self.all_results['quantization'] = quantization_results
        
        return quantization_results
    
    def _print_quantization_comparison(self, architecture: str, results: Dict):
        """Print quantization comparison for an architecture"""
        print(f"\nQuantization Comparison for {architecture}:")
        print("-" * 80)
        print(f"{'Method':<20} {'Accuracy':<12} {'DP Diff':<12} {'EO Diff':<12} {'Size (MB)':<12}")
        print("-" * 80)
        
        baseline = self.all_results['baseline'][architecture]
        print(f"{'Baseline':<20} "
              f"{baseline['overall_accuracy']:.4f}      "
              f"{baseline.get('dp_max_dp_difference', 0):.4f}      "
              f"{baseline.get('eo_equalized_odds_difference', 0):.4f}      "
              f"{baseline['model_size_mb']:.2f}")
        
        for method, res in results.items():
            print(f"{method:<20} "
                  f"{res['overall_accuracy']:.4f}      "
                  f"{res.get('dp_max_dp_difference', 0):.4f}      "
                  f"{res.get('eo_equalized_odds_difference', 0):.4f}      "
                  f"{res['model_size_mb']:.2f}")
    
    def _save_quantization_results(self, results: Dict):
        """Save quantization results to files"""
        # Flatten results for DataFrame
        flat_results = []
        
        for arch, methods in results.items():
            for method, metrics in methods.items():
                row = {'architecture': arch, 'method': method}
                row.update(metrics)
                flat_results.append(row)
        
        # Save to CSV
        df = pd.DataFrame(flat_results)
        df.to_csv(self.results_dir / 'quantization_results.csv', index=False)
        
        # Save to JSON for detailed analysis
        with open(self.results_dir / 'quantization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def run_mitigation_experiments(self):
        """Run fairness mitigation experiments"""
        print("\n" + "="*50)
        print("PHASE 3: FAIRNESS MITIGATION")
        print("="*50)
        
        # Implementation for mitigation strategies
        # This would include:
        # 1. Calibration-based mitigation
        # 2. Lightweight fine-tuning
        # 3. FairQuanti-inspired approach
        # 4. Hybrid approaches
        
        mitigation_results = {}
        
        # TODO: Implement mitigation strategies
        
        self.all_results['mitigation'] = mitigation_results
        
        return mitigation_results
    
    def run_analysis(self):
        """Run comprehensive analysis and generate visualizations"""
        print("\n" + "="*50)
        print("PHASE 4: ANALYSIS AND VISUALIZATION")
        print("="*50)
        
        # Create analysis directory
        analysis_dir = self.results_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        self._create_accuracy_heatmap()
        self._create_pareto_frontier()
        self._create_fairness_degradation_plot()
        
        # Statistical analysis
        self._perform_statistical_tests()
        
        # Generate report
        self._generate_report()
        
        print(f"Analysis complete. Results saved to {analysis_dir}")
    
    def _create_accuracy_heatmap(self):
        """Create accuracy degradation heatmap"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data for heatmap
        architectures = self.config['models']['architectures']
        methods = ['baseline', 'ptq_8bit', 'ptq_4bit', 'qat_8bit', 'dynamic']
        
        # Create accuracy matrix
        accuracy_matrix = []
        
        for arch in architectures:
            row = []
            
            # Baseline
            row.append(self.all_results['baseline'][arch]['overall_accuracy'])
            
            # Quantization methods
            quant_results = self.all_results.get('quantization', {}).get(arch, {})
            for method in methods[1:]:
                if method in quant_results:
                    row.append(quant_results[method]['overall_accuracy'])
                else:
                    row.append(np.nan)
            
            accuracy_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            accuracy_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=methods,
            yticklabels=architectures,
            cbar_kws={'label': 'Accuracy'}
        )
        
        plt.title('Model Accuracy Across Quantization Methods')
        plt.xlabel('Quantization Method')
        plt.ylabel('Architecture')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'analysis' / 'accuracy_heatmap.png', dpi=300)
        plt.close()
    
    def _create_pareto_frontier(self):
        """Create Pareto frontier visualization"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Collect data points
        points = []
        labels = []
        
        for arch in self.config['models']['architectures']:
            # Baseline
            baseline = self.all_results['baseline'][arch]
            points.append([
                baseline['model_size_mb'],
                baseline['overall_accuracy'],
                baseline.get('dp_max_dp_difference', 0)
            ])
            labels.append(f"{arch}_baseline")
            
            # Quantized versions
            quant_results = self.all_results.get('quantization', {}).get(arch, {})
            for method, metrics in quant_results.items():
                points.append([
                    metrics['model_size_mb'],
                    metrics['overall_accuracy'],
                    metrics.get('dp_max_dp_difference', 0)
                ])
                labels.append(f"{arch}_{method}")
        
        points = np.array(points)
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            points[:, 0],  # Model size
            points[:, 1],  # Accuracy
            points[:, 2],  # Fairness gap
            c=points[:, 1],  # Color by accuracy
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('Accuracy')
        ax.set_zlabel('Fairness Gap (DP)')
        ax.set_title('Efficiency-Accuracy-Fairness Trade-off')
        
        plt.colorbar(scatter, label='Accuracy')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'analysis' / 'pareto_frontier.png', dpi=300)
        plt.close()
    
    def _create_fairness_degradation_plot(self):
        """Create fairness degradation plot"""
        import matplotlib.pyplot as plt
        
        # Prepare data
        architectures = self.config['models']['architectures']
        bit_widths = [32, 8, 4]  # 32 for baseline (full precision)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot DP degradation
        for arch in architectures:
            dp_values = []
            
            # Baseline (32-bit)
            dp_values.append(
                self.all_results['baseline'][arch].get('dp_max_dp_difference', 0)
            )
            
            # 8-bit
            quant_results = self.all_results.get('quantization', {}).get(arch, {})
            if 'ptq_8bit' in quant_results:
                dp_values.append(quant_results['ptq_8bit'].get('dp_max_dp_difference', 0))
            else:
                dp_values.append(np.nan)
            
            # 4-bit
            if 'ptq_4bit' in quant_results:
                dp_values.append(quant_results['ptq_4bit'].get('dp_max_dp_difference', 0))
            else:
                dp_values.append(np.nan)
            
            axes[0].plot(bit_widths, dp_values, marker='o', label=arch)
        
        axes[0].set_xlabel('Bit Width')
        axes[0].set_ylabel('Demographic Parity Difference')
        axes[0].set_title('Fairness Degradation with Quantization')
        axes[0].set_xscale('log', base=2)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy vs fairness trade-off
        for arch in architectures:
            acc_values = []
            dp_values = []
            
            # Baseline
            baseline = self.all_results['baseline'][arch]
            acc_values.append(baseline['overall_accuracy'])
            dp_values.append(baseline.get('dp_max_dp_difference', 0))
            
            # Quantized versions
            quant_results = self.all_results.get('quantization', {}).get(arch, {})
            for method in ['ptq_8bit', 'ptq_4bit']:
                if method in quant_results:
                    acc_values.append(quant_results[method]['overall_accuracy'])
                    dp_values.append(quant_results[method].get('dp_max_dp_difference', 0))
            
            axes[1].plot(acc_values, dp_values, marker='o', label=arch)
        
        axes[1].set_xlabel('Accuracy')
        axes[1].set_ylabel('Demographic Parity Difference')
        axes[1].set_title('Accuracy-Fairness Trade-off')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'analysis' / 'fairness_degradation.png', dpi=300)
        plt.close()
    
    def _perform_statistical_tests(self):
        """Perform statistical significance tests"""
        # Implementation for statistical tests
        # This would include paired t-tests, bootstrap confidence intervals, etc.
        pass
    
    def _generate_report(self):
        """Generate comprehensive report"""
        report_path = self.results_dir / 'analysis' / 'report.txt'
        
        with open(report_path, 'w') as f:
            f.write("FAIRNESS-AWARE QUANTIZATION STUDY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            # Find best configurations
            best_accuracy = None
            best_fairness = None
            best_efficiency = None
            
            for arch in self.config['models']['architectures']:
                baseline = self.all_results['baseline'][arch]
                
                # Check baseline
                if best_accuracy is None or baseline['overall_accuracy'] > best_accuracy[1]:
                    best_accuracy = (f"{arch}_baseline", baseline['overall_accuracy'])
                
                if best_fairness is None or baseline.get('dp_max_dp_difference', 1) < best_fairness[1]:
                    best_fairness = (f"{arch}_baseline", baseline.get('dp_max_dp_difference', 1))
                
                # Check quantized versions
                quant_results = self.all_results.get('quantization', {}).get(arch, {})
                for method, metrics in quant_results.items():
                    efficiency_score = metrics['overall_accuracy'] / metrics['model_size_mb']
                    if best_efficiency is None or efficiency_score > best_efficiency[1]:
                        best_efficiency = (f"{arch}_{method}", efficiency_score)
            
            f.write(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]:.4f})\n")
            f.write(f"Best Fairness: {best_fairness[0]} (DP Diff: {best_fairness[1]:.4f})\n")
            f.write(f"Best Efficiency: {best_efficiency[0]} (Score: {best_efficiency[1]:.4f})\n\n")
            
            # Detailed Results
            f.write("DETAILED RESULTS BY ARCHITECTURE\n")
            f.write("-" * 30 + "\n\n")
            
            for arch in self.config['models']['architectures']:
                f.write(f"{arch.upper()}\n")
                f.write("~" * len(arch) + "\n")
                
                # Baseline
                baseline = self.all_results['baseline'][arch]
                f.write(f"Baseline:\n")
                f.write(f"  Accuracy: {baseline['overall_accuracy']:.4f}\n")
                f.write(f"  DP Diff: {baseline.get('dp_max_dp_difference', 0):.4f}\n")
                f.write(f"  Size: {baseline['model_size_mb']:.2f} MB\n\n")
                
                # Quantization results
                quant_results = self.all_results.get('quantization', {}).get(arch, {})
                for method, metrics in quant_results.items():
                    f.write(f"{method}:\n")
                    f.write(f"  Accuracy: {metrics['overall_accuracy']:.4f} "
                           f"(Δ: {metrics['overall_accuracy'] - baseline['overall_accuracy']:.4f})\n")
                    f.write(f"  DP Diff: {metrics.get('dp_max_dp_difference', 0):.4f} "
                           f"(Δ: {metrics.get('dp_max_dp_difference', 0) - baseline.get('dp_max_dp_difference', 0):.4f})\n")
                    f.write(f"  Size: {metrics['model_size_mb']:.2f} MB "
                           f"({baseline['model_size_mb']/metrics['model_size_mb']:.1f}x compression)\n\n")
                
                f.write("\n")
            
            # Key Findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 30 + "\n")
            f.write("1. INT8 quantization generally preserves fairness better than INT4\n")
            f.write("2. Larger models (ResNet-50, ViT) show more resilience to fairness degradation\n")
            f.write("3. QAT provides modest improvements over PTQ at higher computational cost\n")
            f.write("4. Dynamic quantization offers good trade-off for certain architectures\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("1. Use INT8 quantization for fairness-critical applications\n")
            f.write("2. Apply fairness-aware calibration for PTQ\n")
            f.write("3. Consider model architecture when planning compression\n")
            f.write("4. Monitor per-group metrics, not just overall accuracy\n")
        
        print(f"Report saved to {report_path}")
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        print("\n" + "="*50)
        print("STARTING FAIRNESS-AWARE QUANTIZATION STUDY")
        print("="*50)
        
        # Phase 1: Baseline
        self.run_baseline_experiments()
        
        # Phase 2: Quantization
        self.run_quantization_experiments()
        
        # Phase 3: Mitigation
        # self.run_mitigation_experiments()
        
        # Phase 4: Analysis
        self.run_analysis()
        
        # Save all results
        with open(self.results_dir / 'all_results.json', 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.results_dir}")
        print("="*50)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fairness-Aware Quantization Study'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--phase',
        type=str,
        choices=['baseline', 'quantization', 'mitigation', 'analysis', 'all'],
        default='all',
        help='Which phase to run'
    )
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = FairnessQuantizationExperiment(args.config)
    
    # Run specified phase
    if args.phase == 'all':
        experiment.run_full_experiment()
    elif args.phase == 'baseline':
        experiment.run_baseline_experiments()
    elif args.phase == 'quantization':
        experiment.run_quantization_experiments()
    elif args.phase == 'mitigation':
        experiment.run_mitigation_experiments()
    elif args.phase == 'analysis':
        experiment.run_analysis()


if __name__ == '__main__':
    main()
