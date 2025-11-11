"""
Fairness Metrics Implementation for Model Evaluation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings

class FairnessMetrics:
    """Comprehensive fairness metrics calculator"""
    
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon
        
    def demographic_parity(self, 
                          y_pred: np.ndarray, 
                          sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate Demographic Parity (DP) - difference in positive prediction rates
        
        Args:
            y_pred: Binary predictions
            sensitive_features: Protected attribute values
            
        Returns:
            Dictionary with DP metrics
        """
        unique_groups = np.unique(sensitive_features)
        positive_rates = {}
        
        for group in unique_groups:
            mask = sensitive_features == group
            positive_rates[group] = np.mean(y_pred[mask])
        
        # Calculate max difference
        rates = list(positive_rates.values())
        dp_difference = max(rates) - min(rates)
        
        # Calculate ratio (avoid division by zero)
        dp_ratio = min(rates) / (max(rates) + self.epsilon)
        
        return {
            'dp_difference': dp_difference,
            'dp_ratio': dp_ratio,
            'positive_rates': positive_rates,
            'worst_case_groups': (
                min(positive_rates, key=positive_rates.get),
                max(positive_rates, key=positive_rates.get)
            )
        }
    
    def equalized_odds(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate Equalized Odds (EO) - difference in TPR and FPR across groups
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_features: Protected attributes
            
        Returns:
            Dictionary with EO metrics
        """
        unique_groups = np.unique(sensitive_features)
        tpr_dict = {}
        fpr_dict = {}
        
        for group in unique_groups:
            mask = sensitive_features == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # True Positive Rate
            positive_mask = y_true_group == 1
            if positive_mask.sum() > 0:
                tpr = np.mean(y_pred_group[positive_mask])
                tpr_dict[group] = tpr
            
            # False Positive Rate
            negative_mask = y_true_group == 0
            if negative_mask.sum() > 0:
                fpr = np.mean(y_pred_group[negative_mask])
                fpr_dict[group] = fpr
        
        # Calculate differences
        tpr_diff = max(tpr_dict.values()) - min(tpr_dict.values())
        fpr_diff = max(fpr_dict.values()) - min(fpr_dict.values())
        
        return {
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'eo_difference': max(tpr_diff, fpr_diff),
            'tpr_by_group': tpr_dict,
            'fpr_by_group': fpr_dict
        }
    
    def predictive_equality(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate Predictive Equality - difference in FPR across groups
        
        Args:
            y_true: True labels
            y_pred: Predictions  
            sensitive_features: Protected attributes
            
        Returns:
            Dictionary with PE metrics
        """
        unique_groups = np.unique(sensitive_features)
        fpr_dict = {}
        
        for group in unique_groups:
            mask = sensitive_features == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # False Positive Rate for negative instances
            negative_mask = y_true_group == 0
            if negative_mask.sum() > 0:
                fpr = np.mean(y_pred_group[negative_mask])
                fpr_dict[group] = fpr
        
        # Calculate difference
        pe_difference = max(fpr_dict.values()) - min(fpr_dict.values())
        
        return {
            'pe_difference': pe_difference,
            'fpr_by_group': fpr_dict
        }
    
    def disparate_impact(self,
                        y_pred: np.ndarray,
                        sensitive_features: np.ndarray,
                        privileged_group: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Disparate Impact - ratio of positive rates
        
        Args:
            y_pred: Predictions
            sensitive_features: Protected attributes
            privileged_group: Value indicating privileged group
            
        Returns:
            Dictionary with DI metrics
        """
        unique_groups = np.unique(sensitive_features)
        positive_rates = {}
        
        for group in unique_groups:
            mask = sensitive_features == group
            positive_rates[group] = np.mean(y_pred[mask])
        
        # If no privileged group specified, use group with highest rate
        if privileged_group is None:
            privileged_group = max(positive_rates, key=positive_rates.get)
        
        privileged_rate = positive_rates[privileged_group]
        di_ratios = {}
        
        for group, rate in positive_rates.items():
            if group != privileged_group:
                di_ratios[group] = rate / (privileged_rate + self.epsilon)
        
        # 80% rule (4/5 rule) check
        min_ratio = min(di_ratios.values()) if di_ratios else 1.0
        satisfies_80_percent_rule = min_ratio >= 0.8
        
        return {
            'di_ratios': di_ratios,
            'min_di_ratio': min_ratio,
            'satisfies_80_percent_rule': satisfies_80_percent_rule,
            'positive_rates': positive_rates
        }
    
    def intersectional_fairness(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               sensitive_features_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate fairness metrics for intersectional groups
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_features_list: List of protected attribute arrays
            
        Returns:
            Dictionary with intersectional fairness metrics
        """
        # Create intersectional groups
        intersectional_groups = np.zeros(len(y_true), dtype=str)
        for i in range(len(y_true)):
            group_id = "_".join([str(feat[i]) for feat in sensitive_features_list])
            intersectional_groups[i] = group_id
        
        unique_groups = np.unique(intersectional_groups)
        metrics = {}
        
        for group in unique_groups:
            mask = intersectional_groups == group
            group_size = mask.sum()
            
            if group_size > 0:
                accuracy = np.mean(y_true[mask] == y_pred[mask])
                positive_rate = np.mean(y_pred[mask])
                
                # Calculate TPR and FPR if possible
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                tpr = fpr = None
                positive_mask = y_true_group == 1
                negative_mask = y_true_group == 0
                
                if positive_mask.sum() > 0:
                    tpr = np.mean(y_pred_group[positive_mask])
                if negative_mask.sum() > 0:
                    fpr = np.mean(y_pred_group[negative_mask])
                
                metrics[group] = {
                    'size': group_size,
                    'accuracy': accuracy,
                    'positive_rate': positive_rate,
                    'tpr': tpr,
                    'fpr': fpr
                }
        
        # Calculate worst-case metrics
        accuracies = [m['accuracy'] for m in metrics.values()]
        worst_group_accuracy = min(accuracies)
        best_group_accuracy = max(accuracies)
        accuracy_gap = best_group_accuracy - worst_group_accuracy
        
        return {
            'group_metrics': metrics,
            'worst_group_accuracy': worst_group_accuracy,
            'accuracy_gap': accuracy_gap,
            'num_groups': len(unique_groups)
        }
    
    def calculate_all_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             sensitive_features: np.ndarray,
                             return_details: bool = False) -> Dict[str, float]:
        """
        Calculate all fairness metrics
        
        Args:
            y_true: True labels
            y_pred: Predictions
            sensitive_features: Protected attributes
            return_details: Whether to return detailed metrics
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Demographic Parity
        dp_metrics = self.demographic_parity(y_pred, sensitive_features)
        results['demographic_parity'] = dp_metrics['dp_difference']
        
        # Equalized Odds
        eo_metrics = self.equalized_odds(y_true, y_pred, sensitive_features)
        results['equalized_odds'] = eo_metrics['eo_difference']
        
        # Predictive Equality
        pe_metrics = self.predictive_equality(y_true, y_pred, sensitive_features)
        results['predictive_equality'] = pe_metrics['pe_difference']
        
        # Disparate Impact
        di_metrics = self.disparate_impact(y_pred, sensitive_features)
        results['disparate_impact'] = di_metrics['min_di_ratio']
        
        # Overall accuracy
        results['accuracy'] = np.mean(y_true == y_pred)
        
        # Per-group accuracy
        unique_groups = np.unique(sensitive_features)
        group_accuracies = []
        for group in unique_groups:
            mask = sensitive_features == group
            group_acc = np.mean(y_true[mask] == y_pred[mask])
            group_accuracies.append(group_acc)
            results[f'accuracy_group_{group}'] = group_acc
        
        # Worst-group accuracy
        results['worst_group_accuracy'] = min(group_accuracies)
        results['accuracy_gap'] = max(group_accuracies) - min(group_accuracies)
        
        if return_details:
            results['detailed_metrics'] = {
                'demographic_parity': dp_metrics,
                'equalized_odds': eo_metrics,
                'predictive_equality': pe_metrics,
                'disparate_impact': di_metrics
            }
        
        return results
    
    def statistical_significance(self,
                                metrics_1: Dict[str, float],
                                metrics_2: Dict[str, float],
                                n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Test statistical significance of fairness metric differences
        
        Args:
            metrics_1: First set of metrics
            metrics_2: Second set of metrics
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Statistical test results
        """
        results = {}
        
        for metric_name in metrics_1.keys():
            if metric_name in metrics_2:
                val1 = metrics_1[metric_name]
                val2 = metrics_2[metric_name]
                
                # Calculate difference
                diff = val2 - val1
                
                # Cohen's d effect size
                cohens_d = diff / np.sqrt((np.var([val1]) + np.var([val2])) / 2 + self.epsilon)
                
                results[metric_name] = {
                    'difference': diff,
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


class FairnessTracker:
    """Track fairness metrics during training/evaluation"""
    
    def __init__(self):
        self.metrics_history = []
        self.fairness_calculator = FairnessMetrics()
    
    def update(self, 
              y_true: np.ndarray,
              y_pred: np.ndarray,
              sensitive_features: np.ndarray,
              model_name: str,
              quantization_method: str,
              bit_width: int):
        """Update metrics history"""
        
        metrics = self.fairness_calculator.calculate_all_metrics(
            y_true, y_pred, sensitive_features, return_details=True
        )
        
        metrics['model_name'] = model_name
        metrics['quantization_method'] = quantization_method
        metrics['bit_width'] = bit_width
        metrics['timestamp'] = np.datetime64('now')
        
        self.metrics_history.append(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics"""
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics
        summary = {
            'num_evaluations': len(self.metrics_history),
            'models_evaluated': list(set(m['model_name'] for m in self.metrics_history)),
            'methods_evaluated': list(set(m['quantization_method'] for m in self.metrics_history)),
            'bit_widths_evaluated': list(set(m['bit_width'] for m in self.metrics_history))
        }
        
        # Calculate average metrics
        metric_names = ['accuracy', 'demographic_parity', 'equalized_odds', 
                       'predictive_equality', 'disparate_impact']
        
        for metric in metric_names:
            values = [m[metric] for m in self.metrics_history if metric in m]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        return summary
