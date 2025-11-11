"""
Visualization module for fairness-aware quantization results
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class FairnessVisualizer:
    """Visualize fairness metrics and quantization results"""
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = results_dir
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_accuracy_fairness_heatmap(self, 
                                       results: Dict,
                                       save_path: Optional[str] = None):
        """
        Create heatmap showing accuracy degradation across models, groups, and quantization levels
        
        Args:
            results: Dictionary of experimental results
            save_path: Path to save the figure
        """
        # Prepare data for heatmap
        data_matrix = []
        models = []
        configurations = []
        
        for key, result in results.items():
            if 'metrics' in result:
                model = result['metrics'].get('model_name', '')
                method = result['metrics'].get('quantization_method', '')
                bits = result['metrics'].get('bit_width', '')
                accuracy = result['metrics'].get('accuracy', 0)
                
                config_name = f"{method}_{bits}bit"
                
                if model not in models:
                    models.append(model)
                if config_name not in configurations:
                    configurations.append(config_name)
        
        # Create matrix
        matrix = np.zeros((len(models), len(configurations)))
        
        for key, result in results.items():
            if 'metrics' in result:
                model = result['metrics'].get('model_name', '')
                method = result['metrics'].get('quantization_method', '')
                bits = result['metrics'].get('bit_width', '')
                accuracy = result['metrics'].get('accuracy', 0)
                
                config_name = f"{method}_{bits}bit"
                
                if model in models and config_name in configurations:
                    i = models.index(model)
                    j = configurations.index(config_name)
                    matrix[i, j] = accuracy
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(matrix, 
                   xticklabels=configurations,
                   yticklabels=models,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0,
                   vmax=1,
                   ax=ax,
                   cbar_kws={'label': 'Accuracy'})
        
        ax.set_title('Accuracy Across Models and Quantization Configurations', fontsize=16, pad=20)
        ax.set_xlabel('Quantization Configuration', fontsize=12)
        ax.set_ylabel('Model Architecture', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_pareto_frontier(self,
                               pareto_points: List[Dict],
                               save_path: Optional[str] = None):
        """
        Create 3D scatter plot showing Pareto frontier
        
        Args:
            pareto_points: List of Pareto optimal points
            save_path: Path to save the figure
        """
        # Extract data
        accuracies = [p['accuracy'] for p in pareto_points]
        fairness_gaps = [p['fairness_gap'] for p in pareto_points]
        model_sizes = [p['model_size'] for p in pareto_points]
        names = [p['name'] for p in pareto_points]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=model_sizes,
            y=accuracies,
            z=fairness_gaps,
            mode='markers+text',
            marker=dict(
                size=10,
                color=fairness_gaps,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Fairness Gap (DP)")
            ),
            text=names,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Size: %{x:.2f} MB<br>' +
                         'Accuracy: %{y:.3f}<br>' +
                         'Fairness Gap: %{z:.3f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D Pareto Frontier: Efficiency-Accuracy-Fairness Trade-off',
            scene=dict(
                xaxis_title='Model Size (MB)',
                yaxis_title='Accuracy',
                zaxis_title='Fairness Gap (DP)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def plot_fairness_comparison(self,
                                results: Dict,
                                metric: str = 'demographic_parity',
                                save_path: Optional[str] = None):
        """
        Compare fairness metrics across different configurations
        
        Args:
            results: Experimental results
            metric: Fairness metric to compare
            save_path: Path to save figure
        """
        # Prepare data
        data = []
        for key, result in results.items():
            if 'metrics' in result and metric in result['metrics']:
                data.append({
                    'Configuration': key,
                    'Model': result['metrics'].get('model_name', ''),
                    'Method': result['metrics'].get('quantization_method', ''),
                    'Bits': result['metrics'].get('bit_width', 32),
                    metric: result['metrics'][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group by model and method
        pivot_df = df.pivot_table(values=metric, 
                                  index='Model', 
                                  columns=['Method', 'Bits'],
                                  aggfunc='mean')
        
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(f'{metric.replace("_", " ").title()} Across Models and Quantization Methods', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Model Architecture', fontsize=12)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.legend(title='Method_Bits', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for baseline
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_group_performance(self,
                                  results: Dict,
                                  save_path: Optional[str] = None):
        """
        Plot performance for each demographic group
        
        Args:
            results: Experimental results
            save_path: Path to save figure
        """
        # Extract per-group metrics
        group_data = []
        
        for key, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                
                # Look for per-group accuracies
                for metric_key, metric_value in metrics.items():
                    if 'accuracy_group_' in metric_key:
                        group_id = metric_key.replace('accuracy_group_', '')
                        group_data.append({
                            'Configuration': key,
                            'Model': metrics.get('model_name', ''),
                            'Method': metrics.get('quantization_method', ''),
                            'Group': group_id,
                            'Accuracy': metric_value
                        })
        
        if not group_data:
            print("No per-group data found")
            return
        
        df = pd.DataFrame(group_data)
        
        # Create violin plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Violin plot by group
        sns.violinplot(data=df, x='Group', y='Accuracy', hue='Method', ax=axes[0])
        axes[0].set_title('Accuracy Distribution by Demographic Group', fontsize=14)
        axes[0].set_xlabel('Group ID', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        
        # Box plot by model
        sns.boxplot(data=df, x='Model', y='Accuracy', hue='Group', ax=axes[1])
        axes[1].set_title('Accuracy Distribution by Model Architecture', fontsize=14)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_quantization_impact(self,
                                baseline_results: Dict,
                                quantized_results: Dict,
                                save_path: Optional[str] = None):
        """
        Show impact of quantization on accuracy and fairness
        
        Args:
            baseline_results: Baseline model results
            quantized_results: Quantized model results
            save_path: Path to save figure
        """
        # Calculate deltas
        deltas = []
        
        for model_name in set(r['metrics']['model_name'] for r in baseline_results.values() if 'metrics' in r):
            # Find baseline
            baseline = None
            for result in baseline_results.values():
                if 'metrics' in result and result['metrics'].get('model_name') == model_name:
                    baseline = result['metrics']
                    break
            
            if baseline:
                # Find quantized versions
                for key, result in quantized_results.items():
                    if 'metrics' in result and result['metrics'].get('model_name') == model_name:
                        quant = result['metrics']
                        
                        delta = {
                            'Model': model_name,
                            'Method': quant.get('quantization_method', ''),
                            'Bits': quant.get('bit_width', ''),
                            'Accuracy_Delta': quant['accuracy'] - baseline['accuracy'],
                            'DP_Delta': quant['demographic_parity'] - baseline['demographic_parity'],
                            'EO_Delta': quant.get('equalized_odds', 0) - baseline.get('equalized_odds', 0)
                        }
                        deltas.append(delta)
        
        df = pd.DataFrame(deltas)
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Change', 'DP Gap Change', 
                          'Accuracy vs DP Trade-off', 'Method Comparison'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        # Accuracy change
        for i, model in enumerate(df['Model'].unique()):
            model_df = df[df['Model'] == model]
            fig.add_trace(
                go.Bar(name=model, 
                      x=[f"{row['Method']}_{row['Bits']}bit" for _, row in model_df.iterrows()],
                      y=model_df['Accuracy_Delta']),
                row=1, col=1
            )
        
        # DP Gap change
        for i, model in enumerate(df['Model'].unique()):
            model_df = df[df['Model'] == model]
            fig.add_trace(
                go.Bar(name=model,
                      x=[f"{row['Method']}_{row['Bits']}bit" for _, row in model_df.iterrows()],
                      y=model_df['DP_Delta'],
                      showlegend=False),
                row=1, col=2
            )
        
        # Accuracy vs DP trade-off scatter
        fig.add_trace(
            go.Scatter(
                x=df['Accuracy_Delta'],
                y=df['DP_Delta'],
                mode='markers+text',
                text=[f"{row['Model'][:3]}_{row['Bits']}b" for _, row in df.iterrows()],
                textposition="top center",
                marker=dict(size=10, color=df['Bits'], colorscale='Viridis'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Method comparison box plot
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method]
            fig.add_trace(
                go.Box(name=method,
                      y=method_df['Accuracy_Delta'],
                      showlegend=False),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Quantization Impact Analysis",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Configuration", row=1, col=1)
        fig.update_xaxes(title_text="Configuration", row=1, col=2)
        fig.update_xaxes(title_text="Accuracy Change", row=2, col=1)
        fig.update_xaxes(title_text="Quantization Method", row=2, col=2)
        
        fig.update_yaxes(title_text="Δ Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Δ DP Gap", row=1, col=2)
        fig.update_yaxes(title_text="DP Gap Change", row=2, col=1)
        fig.update_yaxes(title_text="Δ Accuracy", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def plot_layer_sensitivity(self,
                              sensitivities: Dict[str, float],
                              save_path: Optional[str] = None):
        """
        Plot layer-wise sensitivity analysis
        
        Args:
            sensitivities: Dictionary of layer sensitivities
            save_path: Path to save figure
        """
        # Sort layers by sensitivity
        sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 layers
        top_layers = sorted_layers[:20]
        
        layers = [l[0].split('.')[-1] for l in top_layers]  # Simplify layer names
        values = [l[1] for l in top_layers]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(layers, values, color=self.color_palette[0])
        
        # Color bars by sensitivity level
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.5:
                bar.set_color('red')
            elif val > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        ax.set_xlabel('Sensitivity Score', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title('Top 20 Most Sensitive Layers to Quantization', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add vertical line for threshold
        ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium Sensitivity')
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='High Sensitivity')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self,
                                  all_results: Dict,
                                  output_dir: str = './visualizations'):
        """
        Create comprehensive visualization report
        
        Args:
            all_results: All experimental results
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Accuracy-Fairness Heatmap
        print("Creating accuracy-fairness heatmap...")
        self.plot_accuracy_fairness_heatmap(
            all_results,
            save_path=os.path.join(output_dir, 'accuracy_heatmap.png')
        )
        
        # 2. Fairness Comparison
        print("Creating fairness comparison plots...")
        for metric in ['demographic_parity', 'equalized_odds']:
            self.plot_fairness_comparison(
                all_results,
                metric=metric,
                save_path=os.path.join(output_dir, f'{metric}_comparison.png')
            )
        
        # 3. Per-group Performance
        print("Creating per-group performance plots...")
        self.plot_per_group_performance(
            all_results,
            save_path=os.path.join(output_dir, 'per_group_performance.png')
        )
        
        # 4. Pareto Frontier (if available)
        if 'pareto_frontier' in all_results:
            print("Creating Pareto frontier visualization...")
            self.plot_3d_pareto_frontier(
                all_results['pareto_frontier'],
                save_path=os.path.join(output_dir, 'pareto_frontier.html')
            )
        
        print(f"All visualizations saved to {output_dir}")


def generate_latex_tables(results: Dict, output_dir: str = './tables'):
    """
    Generate LaTeX tables for paper
    
    Args:
        results: Experimental results
        output_dir: Directory to save tables
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary table
    summary_data = []
    for key, result in results.items():
        if 'metrics' in result:
            summary_data.append({
                'Model': result['metrics'].get('model_name', ''),
                'Method': result['metrics'].get('quantization_method', ''),
                'Bits': result['metrics'].get('bit_width', ''),
                'Accuracy': f"{result['metrics'].get('accuracy', 0):.3f}",
                'DP Gap': f"{result['metrics'].get('demographic_parity', 0):.3f}",
                'EO Gap': f"{result['metrics'].get('equalized_odds', 0):.3f}",
                'Size (MB)': f"{result.get('model_size', {}).get('total_size_mb', 0):.1f}"
            })
    
    df = pd.DataFrame(summary_data)
    
    # Generate LaTeX table
    latex_table = df.to_latex(
        index=False,
        caption='Quantization Impact on Accuracy and Fairness Metrics',
        label='tab:quantization_results',
        column_format='lllrrrr',
        escape=False
    )
    
    # Save to file
    with open(os.path.join(output_dir, 'summary_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX tables saved to {output_dir}")


if __name__ == '__main__':
    # Example usage
    import pickle
    
    # Load sample results (you would load your actual results)
    # with open('results/exp_20240101_120000/phase4_analysis.pkl', 'rb') as f:
    #     results = pickle.load(f)
    
    # Create visualizer
    visualizer = FairnessVisualizer()
    
    # Generate sample data for demonstration
    sample_results = {
        'celeba_resnet50_baseline': {
            'metrics': {
                'model_name': 'resnet50',
                'quantization_method': 'baseline',
                'bit_width': 32,
                'accuracy': 0.95,
                'demographic_parity': 0.05,
                'equalized_odds': 0.04
            },
            'model_size': {'total_size_mb': 97.5}
        },
        'celeba_resnet50_ptq_8': {
            'metrics': {
                'model_name': 'resnet50',
                'quantization_method': 'ptq_static',
                'bit_width': 8,
                'accuracy': 0.93,
                'demographic_parity': 0.08,
                'equalized_odds': 0.07
            },
            'model_size': {'total_size_mb': 24.4}
        }
    }
    
    # Create visualizations
    print("Creating sample visualizations...")
    visualizer.plot_accuracy_fairness_heatmap(sample_results)
