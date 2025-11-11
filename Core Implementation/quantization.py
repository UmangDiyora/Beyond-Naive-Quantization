"""
Quantization methods: PTQ, QAT, Mixed Precision, and Fairness-Aware Quantization
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import quantize_dynamic, quantize, prepare_qat, convert, prepare
from torch.nn.intrinsic import ConvBnReLU2d, ConvBn2d, LinearReLU
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import copy
import warnings
from tqdm import tqdm


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Support both dictionary-based batches ({'image', 'label', 'sensitive'})
    and tuple-based batches (image, label, sensitive).
    """
    if isinstance(batch, dict):
        images = batch.get('image')
        labels = batch.get('label')
        sensitive = batch.get('sensitive')
        return images, labels, sensitive

    if isinstance(batch, (list, tuple)):
        n = len(batch)
        images = batch[0] if n > 0 else None
        labels = batch[1] if n > 1 else None
        sensitive = batch[2] if n > 2 else None
        return images, labels, sensitive

    raise TypeError(f"Unsupported batch type: {type(batch)}")


class ModelQuantizer:
    """
    Thin wrapper to provide the interface expected by main.py:
      - post_training_quantization(model, calibration_loader, bit_width, calibration_method)
      - quantization_aware_training(model, train_loader, val_loader, epochs, learning_rate)
      - dynamic_quantization(model)
    """
    def __init__(self, model: nn.Module, device: str = 'cpu', backend: str = 'fbgemm'):
        self.device = device
        self.backend = backend
        torch.backends.quantized.engine = backend

    def post_training_quantization(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        bit_width: int = 8,
        calibration_method: str = 'minmax',
        n_calibration_batches: int = 10
    ) -> nn.Module:
        ptq = PostTrainingQuantization(bit_width=bit_width, backend=self.backend, calibration_method=calibration_method)
        return ptq.quantize(model, calibration_loader, n_calibration_batches=n_calibration_batches)

    def quantization_aware_training(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        learning_rate: float = 1e-4
    ) -> nn.Module:
        qat = QuantizationAwareTraining(bit_width=8, backend=self.backend)
        qat_model = qat.prepare_qat(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
        device = self.device if torch.cuda.is_available() else 'cpu'
        return qat.train_qat(qat_model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)

    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        # Apply dynamic quantization to Linear layers
        dq_model = copy.deepcopy(model)
        dq_model.eval()
        return quantize_dynamic(dq_model, {nn.Linear}, dtype=torch.qint8)

class QuantizationMethod:
    """Base class for quantization methods"""
    
    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width
        self.qconfig = self._get_qconfig()
    
    def _get_qconfig(self):
        """Get quantization configuration based on bit width"""
        if self.bit_width == 8:
            return torch.quantization.get_default_qconfig('fbgemm')
        elif self.bit_width == 4:
            # Custom 4-bit quantization config
            return torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                )
            )
        else:
            warnings.warn(f"Bit width {self.bit_width} not fully supported, using INT8")
            return torch.quantization.get_default_qconfig('fbgemm')
    
    def quantize(self, model: nn.Module, *args, **kwargs):
        """Quantize the model"""
        raise NotImplementedError
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': size_mb
        }


class PostTrainingQuantization(QuantizationMethod):
    """Post-Training Quantization (PTQ)"""
    
    def __init__(self, 
                 bit_width: int = 8,
                 backend: str = 'fbgemm',
                 calibration_method: str = 'minmax'):
        """
        PTQ initialization
        
        Args:
            bit_width: Quantization bit width
            backend: Quantization backend ('fbgemm' or 'qnnpack')
            calibration_method: Calibration method ('minmax', 'histogram', 'entropy')
        """
        super().__init__(bit_width)
        self.backend = backend
        self.calibration_method = calibration_method
        
        # Set backend
        torch.backends.quantized.engine = backend
    
    def quantize(self, 
                 model: nn.Module, 
                 calibration_loader: torch.utils.data.DataLoader,
                 n_calibration_batches: int = 10) -> nn.Module:
        """
        Perform post-training quantization
        
        Args:
            model: Model to quantize
            calibration_loader: DataLoader for calibration
            n_calibration_batches: Number of batches for calibration
            
        Returns:
            Quantized model
        """
        # Clone model
        quantized_model = copy.deepcopy(model)
        quantized_model.eval()
        
        # Fuse modules (Conv+BN+ReLU, etc.)
        quantized_model = self._fuse_modules(quantized_model)
        
        # Prepare for quantization
        quantized_model.qconfig = self.qconfig
        prepare(quantized_model, inplace=True)
        
        # Calibration
        print(f"Calibrating with {n_calibration_batches} batches...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= n_calibration_batches:
                    break
                inputs, _, _ = _unpack_batch(batch)
                if inputs is None:
                    continue
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True)
                    quantized_model = quantized_model.cuda()
                quantized_model(inputs)
        
        # Convert to quantized model
        convert(quantized_model, inplace=True)
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for better quantization"""
        # This is model-specific, implementing for common architectures
        model_name = model.__class__.__name__.lower()
        
        if 'resnet' in model_name:
            # ResNet fusion
            modules_to_fuse = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Look for patterns like conv-bn-relu
                    modules_to_fuse.append([name])
            
            if modules_to_fuse:
                torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)
        
        return model


class QuantizationAwareTraining(QuantizationMethod):
    """Quantization-Aware Training (QAT)"""
    
    def __init__(self, 
                 bit_width: int = 8,
                 backend: str = 'fbgemm'):
        """
        QAT initialization
        
        Args:
            bit_width: Quantization bit width
            backend: Quantization backend
        """
        super().__init__(bit_width)
        self.backend = backend
        torch.backends.quantized.engine = backend
    
    def prepare_qat(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for QAT
        
        Args:
            model: Model to prepare
            
        Returns:
            Model prepared for QAT
        """
        # Clone model
        qat_model = copy.deepcopy(model)
        
        # Fuse modules
        qat_model = self._fuse_modules(qat_model)
        
        # Set qconfig
        qat_model.qconfig = self.qconfig
        
        # Prepare for QAT
        qat_model.train()
        prepare_qat(qat_model, inplace=True)
        
        return qat_model
    
    def train_qat(self,
                  model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  num_epochs: int = 5,
                  device: str = 'cuda') -> nn.Module:
        """
        Perform QAT training
        
        Args:
            model: Prepared QAT model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of training epochs
            device: Device to train on
            
        Returns:
            Trained quantized model
        """
        model = model.to(device)
        best_acc = 0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'QAT Epoch {epoch+1}/{num_epochs}')
            for batch in pbar:
                inputs, labels, _ = _unpack_batch(batch)
                if inputs is None or labels is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/(pbar.n+1):.3f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels, _ = _unpack_batch(batch)
                    if inputs is None or labels is None:
                        continue
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            print(f'Epoch {epoch+1}: Val Acc = {val_acc:.2f}%')
            
            if val_acc > best_acc:
                best_acc = val_acc
        
        # Convert to fully quantized model
        model.eval()
        quantized_model = convert(model)
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for QAT"""
        # Similar to PTQ fusion
        return model


class MixedPrecisionQuantization(QuantizationMethod):
    """Mixed Precision Quantization with layer-wise bit allocation"""
    
    def __init__(self, 
                 bit_widths: Dict[str, int] = None,
                 sensitivity_analysis: bool = True):
        """
        Mixed precision initialization
        
        Args:
            bit_widths: Dictionary mapping layer names to bit widths
            sensitivity_analysis: Whether to perform sensitivity analysis
        """
        super().__init__(8)  # Default bit width
        self.bit_widths = bit_widths or {}
        self.sensitivity_analysis = sensitivity_analysis
        self.layer_sensitivities = {}
    
    def analyze_layer_sensitivity(self,
                                 model: nn.Module,
                                 val_loader: torch.utils.data.DataLoader,
                                 criterion: nn.Module,
                                 device: str = 'cuda') -> Dict[str, float]:
        """
        Analyze sensitivity of each layer to quantization
        
        Args:
            model: Model to analyze
            val_loader: Validation data loader
            criterion: Loss criterion
            device: Device
            
        Returns:
            Dictionary of layer sensitivities
        """
        model = model.to(device)
        model.eval()
        
        # Get baseline performance
        baseline_loss = self._evaluate_model(model, val_loader, criterion, device)
        
        sensitivities = {}
        
        # Test each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Temporarily quantize this layer
                temp_model = copy.deepcopy(model)
                self._quantize_layer(temp_model, name, bit_width=4)
                
                # Evaluate impact
                new_loss = self._evaluate_model(temp_model, val_loader, criterion, device)
                sensitivity = (new_loss - baseline_loss) / baseline_loss
                sensitivities[name] = sensitivity
                
                del temp_model
        
        self.layer_sensitivities = sensitivities
        return sensitivities
    
    def allocate_bits_by_sensitivity(self,
                                    total_bits_budget: int,
                                    min_bits: int = 2,
                                    max_bits: int = 8) -> Dict[str, int]:
        """
        Allocate bits to layers based on sensitivity
        
        Args:
            total_bits_budget: Total bit budget
            min_bits: Minimum bits per layer
            max_bits: Maximum bits per layer
            
        Returns:
            Bit allocation dictionary
        """
        if not self.layer_sensitivities:
            raise ValueError("Run sensitivity analysis first")
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(self.layer_sensitivities.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Allocate bits proportionally
        bit_allocation = {}
        remaining_budget = total_bits_budget
        n_layers = len(sorted_layers)
        
        for i, (layer_name, sensitivity) in enumerate(sorted_layers):
            # More sensitive layers get more bits
            if i < n_layers // 3:  # Top 1/3 most sensitive
                bits = max_bits
            elif i < 2 * n_layers // 3:  # Middle 1/3
                bits = (min_bits + max_bits) // 2
            else:  # Bottom 1/3
                bits = min_bits
            
            bit_allocation[layer_name] = bits
            remaining_budget -= bits
        
        self.bit_widths = bit_allocation
        return bit_allocation
    
    def quantize(self,
                 model: nn.Module,
                 calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        Apply mixed precision quantization
        
        Args:
            model: Model to quantize
            calibration_loader: Calibration data
            
        Returns:
            Mixed precision quantized model
        """
        quantized_model = copy.deepcopy(model)
        
        # Apply bit widths to each layer
        for layer_name, bit_width in self.bit_widths.items():
            self._quantize_layer(quantized_model, layer_name, bit_width)
        
        return quantized_model
    
    def _quantize_layer(self, model: nn.Module, layer_name: str, bit_width: int):
        """Quantize a specific layer"""
        # Implementation depends on specific quantization backend
        # This is a simplified version
        for name, module in model.named_modules():
            if name == layer_name:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Apply quantization to this layer
                    module.qconfig = self._get_qconfig_for_bits(bit_width)
    
    def _get_qconfig_for_bits(self, bit_width: int):
        """Get qconfig for specific bit width"""
        if bit_width == 8:
            return torch.quantization.get_default_qconfig('fbgemm')
        elif bit_width == 4:
            return torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8, reduce_range=True
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8, reduce_range=True
                )
            )
        else:
            return torch.quantization.get_default_qconfig('fbgemm')
    
    def _evaluate_model(self,
                       model: nn.Module,
                       val_loader: torch.utils.data.DataLoader,
                       criterion: nn.Module,
                       device: str) -> float:
        """Evaluate model performance"""
        model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, _ = _unpack_batch(batch)
                if inputs is None or labels is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches


class FairnessAwareQuantization(QuantizationMethod):
    """Fairness-aware quantization methods"""
    
    def __init__(self,
                 base_method: str = 'ptq',
                 fairness_constraint: str = 'demographic_parity',
                 fairness_weight: float = 0.1):
        """
        Fairness-aware quantization
        
        Args:
            base_method: Base quantization method ('ptq' or 'qat')
            fairness_constraint: Fairness metric to optimize
            fairness_weight: Weight for fairness loss
        """
        super().__init__()
        self.base_method = base_method
        self.fairness_constraint = fairness_constraint
        self.fairness_weight = fairness_weight
    
    def bias_aware_calibration(self,
                              model: nn.Module,
                              calibration_loader: torch.utils.data.DataLoader,
                              sensitive_groups: List[int]) -> nn.Module:
        """
        Perform bias-aware calibration for PTQ
        
        Args:
            model: Model to quantize
            calibration_loader: Calibration data with balanced groups
            sensitive_groups: List of sensitive group indices
            
        Returns:
            Quantized model with bias-aware calibration
        """
        quantized_model = copy.deepcopy(model)
        quantized_model.eval()
        
        # Prepare for quantization with group-specific calibration
        quantized_model.qconfig = self.qconfig
        prepare(quantized_model, inplace=True)
        
        # Calibrate with emphasis on underrepresented groups
        print("Performing bias-aware calibration...")
        with torch.no_grad():
            for batch in calibration_loader:
                inputs, _, sensitive = _unpack_batch(batch)
                if inputs is None or sensitive is None:
                    continue
                
                # Give more weight to underrepresented groups
                for group_id in sensitive_groups:
                    group_mask = sensitive == group_id
                    if group_mask.any():
                        group_inputs = inputs[group_mask]
                        if torch.cuda.is_available():
                            group_inputs = group_inputs.cuda()
                            quantized_model = quantized_model.cuda()
                        
                        # Run multiple times for underrepresented groups
                        for _ in range(2):  # Double weight
                            quantized_model(group_inputs)
        
        # Convert to quantized model
        convert(quantized_model, inplace=True)
        
        return quantized_model
    
    def fairness_constrained_finetuning(self,
                                       model: nn.Module,
                                       train_loader: torch.utils.data.DataLoader,
                                       criterion: nn.Module,
                                       fairness_metric: Callable,
                                       optimizer: torch.optim.Optimizer,
                                       num_epochs: int = 2,
                                       device: str = 'cuda') -> nn.Module:
        """
        Fine-tune with fairness constraints
        
        Args:
            model: Quantized model to fine-tune
            train_loader: Training data
            criterion: Main loss function
            fairness_metric: Fairness metric function
            optimizer: Optimizer
            num_epochs: Number of epochs
            device: Device
            
        Returns:
            Fine-tuned model
        """
        model = model.to(device)
        
        for epoch in range(num_epochs):
            model.train()
            
            pbar = tqdm(train_loader, desc=f'Fairness Fine-tuning {epoch+1}/{num_epochs}')
            for batch in pbar:
                inputs, labels, sensitive = _unpack_batch(batch)
                if inputs is None or labels is None or sensitive is None:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                sensitive = sensitive.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Main loss
                main_loss = criterion(outputs, labels)
                
                # Fairness loss
                fairness_loss = self._compute_fairness_loss(
                    outputs, labels, sensitive, fairness_metric
                )
                
                # Combined loss
                total_loss = main_loss + self.fairness_weight * fairness_loss
                
                total_loss.backward()
                optimizer.step()
                
                pbar.set_postfix({
                    'main_loss': f'{main_loss.item():.3f}',
                    'fair_loss': f'{fairness_loss.item():.3f}'
                })
        
        return model
    
    def identify_bias_sensitive_neurons(self,
                                       model: nn.Module,
                                       val_loader: torch.utils.data.DataLoader,
                                       device: str = 'cuda') -> Dict[str, List[int]]:
        """
        Identify neurons that are sensitive to bias
        
        Args:
            model: Model to analyze
            val_loader: Validation data
            device: Device
            
        Returns:
            Dictionary mapping layer names to sensitive neuron indices
        """
        model = model.to(device)
        model.eval()
        
        sensitive_neurons = {}
        
        # Hook to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Collect activations for different groups
        group_activations = {}
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, _, sensitive = _unpack_batch(batch)
                if inputs is None or sensitive is None:
                    continue
                inputs = inputs.to(device)
                sensitive = sensitive.to(device)
                
                outputs = model(inputs)
                
                # Store activations by group
                for group_id in torch.unique(sensitive):
                    group_mask = sensitive == group_id
                    if not group_mask.any():
                        continue
                    group_key = group_id.item()
                    if group_key not in group_activations:
                        group_activations[group_key] = {}
                    
                    for name, act in activations.items():
                        if name not in group_activations[group_key]:
                            group_activations[group_key][name] = []
                        group_activations[group_key][name].append(
                            act[group_mask].mean(dim=0).cpu()
                        )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Identify neurons with high variance across groups
        for layer_name in activations.keys():
            layer_sensitive = []
            
            # Calculate variance across groups for each neuron
            group_means = []
            for group_id in group_activations.keys():
                if layer_name in group_activations[group_id]:
                    mean_act = torch.stack(group_activations[group_id][layer_name]).mean(0)
                    group_means.append(mean_act)
            
            if len(group_means) > 1:
                group_means = torch.stack(group_means)
                variance = group_means.var(dim=0)
                
                # Find high-variance neurons (top 20%)
                threshold = torch.quantile(variance.flatten(), 0.8)
                sensitive_mask = variance > threshold
                
                # Get indices of sensitive neurons
                if len(variance.shape) == 3:  # Conv layer
                    sensitive_indices = torch.where(sensitive_mask)[0].tolist()
                else:  # Linear layer
                    sensitive_indices = torch.where(sensitive_mask)[0].tolist()
                
                sensitive_neurons[layer_name] = sensitive_indices
        
        return sensitive_neurons
    
    def _compute_fairness_loss(self,
                              outputs: torch.Tensor,
                              labels: torch.Tensor,
                              sensitive: torch.Tensor,
                              fairness_metric: Callable) -> torch.Tensor:
        """Compute fairness loss"""
        # Simplified fairness loss based on demographic parity
        predictions = outputs.argmax(dim=1)
        
        # Calculate positive rates for each group
        unique_groups = torch.unique(sensitive)
        positive_rates = []
        
        for group_id in unique_groups:
            group_mask = sensitive == group_id
            group_preds = predictions[group_mask]
            positive_rate = (group_preds == 1).float().mean()
            positive_rates.append(positive_rate)
        
        # Fairness loss is variance of positive rates
        if len(positive_rates) > 1:
            rates = torch.stack(positive_rates)
            fairness_loss = rates.var()
        else:
            fairness_loss = torch.tensor(0.0, device=outputs.device)
        
        return fairness_loss


def get_quantization_method(method_name: str,
                           bit_width: int = 8,
                           **kwargs) -> QuantizationMethod:
    """
    Factory function to get quantization method
    
    Args:
        method_name: Name of quantization method
        bit_width: Bit width for quantization
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Quantization method instance
    """
    methods = {
        'ptq_static': PostTrainingQuantization,
        'ptq_dynamic': PostTrainingQuantization,
        'qat': QuantizationAwareTraining,
        'mixed_precision': MixedPrecisionQuantization,
        'fairness_aware': FairnessAwareQuantization
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown quantization method: {method_name}")
    
    return methods[method_name](bit_width=bit_width, **kwargs)
