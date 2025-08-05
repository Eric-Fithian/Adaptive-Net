# Comprehensive State-Action Correlation Analysis Framework

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import get_all_datasets, get_dataset_by_name
from copy import deepcopy
from torch.utils.data import DataLoader

class StateMetrics:
    """Comprehensive collection of all possible state measurements"""
    
    @staticmethod
    def compute_all_node_metrics(layer: nn.Linear, activations: torch.Tensor, 
                                gradients: torch.Tensor) -> Dict[str, np.ndarray]:
        """Compute every conceivable node-level metric"""
        
        weights = layer.weight.data  # Shape: [out_features, in_features]
        bias = layer.bias.data if layer.bias is not None else None
        
        metrics = {}
        
        # === WEIGHT-BASED METRICS ===
        metrics['weight_magnitude'] = torch.norm(weights, dim=1).numpy()
        metrics['weight_l1_norm'] = torch.norm(weights, p=1, dim=1).numpy()
        metrics['weight_l2_norm'] = torch.norm(weights, p=2, dim=1).numpy()
        metrics['weight_mean'] = torch.mean(weights, dim=1).numpy()
        metrics['weight_std'] = torch.std(weights, dim=1).numpy()
        metrics['weight_variance'] = torch.var(weights, dim=1).numpy()
        metrics['weight_sparsity'] = (weights == 0).float().mean(dim=1).numpy()
        metrics['weight_rank'] = torch.linalg.matrix_rank(weights.unsqueeze(0)).numpy()
        
        # Weight distribution properties
        metrics['weight_skewness'] = torch.tensor([torch.distributions.utils.lazy_property(
            lambda: torch.pow(weights[i] - weights[i].mean(), 3).mean() / torch.pow(weights[i].std(), 3)
        ) for i in range(weights.shape[0])]).numpy()
        
        # === GRADIENT-BASED METRICS ===
        if gradients is not None:
            metrics['grad_magnitude'] = torch.norm(gradients, dim=1).numpy()
            metrics['grad_variance'] = torch.var(gradients, dim=1).numpy()
            metrics['grad_snr'] = (torch.mean(gradients, dim=1) / (torch.std(gradients, dim=1) + 1e-8)).numpy()
            metrics['grad_direction_stability'] = torch.tensor([
                torch.cosine_similarity(gradients[i, :-1], gradients[i, 1:], dim=0).mean()
                for i in range(gradients.shape[0])
            ]).numpy()
            
            # Gradient-weight interaction
            metrics['weight_grad_correlation'] = torch.tensor([
                torch.corrcoef(torch.stack([weights[i], gradients[i]]))[0, 1]
                for i in range(weights.shape[0])
            ]).numpy()
        
        # === ACTIVATION-BASED METRICS ===
        if activations is not None:
            metrics['activation_mean'] = torch.mean(activations, dim=0).numpy()
            metrics['activation_variance'] = torch.var(activations, dim=0).numpy()
            metrics['activation_sparsity'] = (activations == 0).float().mean(dim=0).numpy()
            metrics['activation_saturation'] = (activations >= 0.99).float().mean(dim=0).numpy()
            metrics['activation_utilization'] = (activations > 0).float().mean(dim=0).numpy()
            
            # Dead neuron detection
            metrics['is_dead'] = (activations == 0).all(dim=0).float().numpy()
            metrics['dying_ratio'] = (activations == 0).float().mean(dim=0).numpy()
        
        # === INFORMATION THEORY METRICS ===
        # Fisher Information approximation
        if gradients is not None:
            metrics['fisher_information'] = torch.mean(gradients**2, dim=1).numpy()
        
        # === STABILITY METRICS ===
        # Weight change rate (would need previous weights)
        # Learning progress (would need historical loss)
        
        return metrics
    
    @staticmethod 
    def compute_layer_metrics(layer: nn.Linear, node_metrics: Dict) -> Dict[str, float]:
        """Aggregate node metrics to layer level"""
        layer_metrics = {}
        
        for metric_name, values in node_metrics.items():
            if len(values) > 0:
                layer_metrics[f'layer_{metric_name}_mean'] = np.mean(values)
                layer_metrics[f'layer_{metric_name}_max'] = np.max(values)
                layer_metrics[f'layer_{metric_name}_min'] = np.min(values)
                layer_metrics[f'layer_{metric_name}_std'] = np.std(values)
                layer_metrics[f'layer_{metric_name}_range'] = np.max(values) - np.min(values)
        
        # Layer-specific metrics
        weights = layer.weight.data
        layer_metrics['layer_condition_number'] = torch.linalg.cond(weights).item()
        layer_metrics['layer_spectral_norm'] = torch.linalg.matrix_norm(weights, ord=2).item()
        layer_metrics['layer_frobenius_norm'] = torch.linalg.matrix_norm(weights, ord='fro').item()
        
        return layer_metrics

class ActionEvaluator:
    """Evaluate the effectiveness of different adaptation actions"""
    
    @staticmethod
    def evaluate_split_action(model: nn.Module, layer_idx: int, neuron_idx: int, 
                            train_loader, val_loader, num_training_steps: int = 100) -> float:
        """Evaluate benefit of splitting a specific neuron"""
        
        # Get baseline performance
        baseline_loss = ActionEvaluator._evaluate_model(model, val_loader)
        
        # Create modified model with split neuron
        modified_model = ActionEvaluator._split_neuron(model, layer_idx, neuron_idx)
        
        # Train for a few steps
        ActionEvaluator._train_briefly(modified_model, train_loader, num_training_steps)
        
        # Measure new performance
        new_loss = ActionEvaluator._evaluate_model(modified_model, val_loader)
        
        # Return improvement (positive = better)
        return baseline_loss - new_loss
    
    @staticmethod
    def evaluate_prune_action(model: nn.Module, layer_idx: int, neuron_idx: int,
                            train_loader, val_loader, num_training_steps: int = 100) -> float:
        """Evaluate benefit of pruning a specific neuron"""
        
        baseline_loss = ActionEvaluator._evaluate_model(model, val_loader)
        modified_model = ActionEvaluator._prune_neuron(model, layer_idx, neuron_idx)
        ActionEvaluator._train_briefly(modified_model, train_loader, num_training_steps)
        new_loss = ActionEvaluator._evaluate_model(modified_model, val_loader)
        
        return baseline_loss - new_loss
    
    @staticmethod
    def evaluate_birth_action(model: nn.Module, layer_idx: int, 
                            train_loader, val_loader, num_training_steps: int = 100,
                            initialization: str = 'random') -> float:
        """Evaluate benefit of adding a new neuron"""
        
        baseline_loss = ActionEvaluator._evaluate_model(model, val_loader)
        modified_model = ActionEvaluator._add_neuron(model, layer_idx, initialization)
        ActionEvaluator._train_briefly(modified_model, train_loader, num_training_steps)
        new_loss = ActionEvaluator._evaluate_model(modified_model, val_loader)
        
        return baseline_loss - new_loss
    
    @staticmethod
    def evaluate_no_action(model: nn.Module, train_loader, val_loader, 
                          num_training_steps: int = 100) -> float:
        """Baseline: just continue training without adaptation"""
        
        baseline_loss = ActionEvaluator._evaluate_model(model, val_loader)
        ActionEvaluator._train_briefly(model, train_loader, num_training_steps)
        new_loss = ActionEvaluator._evaluate_model(model, val_loader)
        
        return baseline_loss - new_loss
    
    # Helper methods (implementation details)
    @staticmethod
    def _evaluate_model(model: nn.Module, val_loader) -> float:
        """Quick validation loss evaluation"""
        model.eval()
        total_loss = 0
        criterion = nn.MSELoss()  # Adjust based on task
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 10:  # Quick eval, don't use full dataset
                    break
                output = model(data)
                total_loss += criterion(output, target).item()
        
        return total_loss / min(10, len(val_loader))
    
    @staticmethod
    def _train_briefly(model: nn.Module, train_loader, num_steps: int):
        """Brief training to measure adaptation benefit"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        step_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if step_count >= num_steps:
                break
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            step_count += 1
    
    @staticmethod
    def _split_neuron(model: nn.Module, layer_idx: int, neuron_idx: int) -> nn.Module:
        """Implementation of neuron splitting (Net2Net style duplication)"""
        # This would duplicate the neuron and divide weights by 2
        # Implementation depends on your specific split strategy
        modified_model = deepcopy(model)
        # ... splitting logic ...
        return modified_model
    
    @staticmethod
    def _prune_neuron(model: nn.Module, layer_idx: int, neuron_idx: int) -> nn.Module:
        """Implementation of neuron pruning"""
        # Zero out the neuron or actually remove it
        modified_model = deepcopy(model)
        # ... pruning logic ...
        return modified_model
    
    @staticmethod
    def _add_neuron(model: nn.Module, layer_idx: int, initialization: str) -> nn.Module:
        """Implementation of neuron addition"""
        # Add new neuron with specified initialization
        modified_model = deepcopy(model)
        # ... addition logic ...
        return modified_model

class CorrelationAnalyzer:
    """Run comprehensive correlation analysis"""
    
    def __init__(self, models: List[nn.Module], datasets: List[Tuple[DataLoader, DataLoader]], 
                 correlation_methods: List[str] = ['pearson', 'spearman']):
        self.models = models
        self.datasets = datasets
        self.correlation_methods = correlation_methods
        self.results = []
    
    def run_full_analysis(self) -> pd.DataFrame:
        """Run correlation analysis across all models, datasets, features, and actions"""
        
        for model_idx, model in enumerate(self.models):
            for dataset_idx, (train_loader, val_loader) in enumerate(self.datasets):
                
                print(f"Analyzing Model {model_idx}, Dataset {dataset_idx}")
                
                # Get all state metrics for each neuron
                state_data = self._collect_state_metrics(model, train_loader)
                
                # Evaluate all actions for each neuron
                action_data = self._evaluate_all_actions(model, train_loader, val_loader)
                
                # Compute correlations
                correlations = self._compute_correlations(state_data, action_data)
                
                # Store results
                for result in correlations:
                    result.update({
                        'model_idx': model_idx,
                        'dataset_idx': dataset_idx
                    })
                    self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def _collect_state_metrics(self, model: nn.Module, train_loader: DataLoader) -> Dict:
        """Collect all state metrics for all neurons"""
        # Run forward/backward pass to get activations and gradients
        # Compute all metrics using StateMetrics class
        # Return dict mapping (layer_idx, neuron_idx) to metric values
        pass
    
    def _evaluate_all_actions(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Evaluate effectiveness of all actions for all neurons"""
        action_results = {}
        
        for layer_idx, layer in enumerate(model.children()):
            if isinstance(layer, nn.Linear):
                for neuron_idx in range(layer.out_features):
                    
                    action_results[(layer_idx, neuron_idx)] = {
                        'split_benefit': ActionEvaluator.evaluate_split_action(
                            model, layer_idx, neuron_idx, train_loader, val_loader),
                        'prune_benefit': ActionEvaluator.evaluate_prune_action(
                            model, layer_idx, neuron_idx, train_loader, val_loader),
                        'birth_benefit': ActionEvaluator.evaluate_birth_action(
                            model, layer_idx, train_loader, val_loader),
                        'no_action_benefit': ActionEvaluator.evaluate_no_action(
                            model, train_loader, val_loader)
                    }
        
        return action_results
    
    def _compute_correlations(self, state_data: Dict, action_data: Dict) -> List[Dict]:
        """Compute correlations between state features and action benefits"""
        correlations = []
        
        # Extract all unique state features
        all_features = set()
        for metrics in state_data.values():
            all_features.update(metrics.keys())
        
        # Extract all action types
        action_types = ['split_benefit', 'prune_benefit', 'birth_benefit', 'no_action_benefit']
        
        # Compute correlation for each feature-action pair
        for feature in all_features:
            for action in action_types:
                
                # Collect data points
                feature_values = []
                action_values = []
                
                for neuron_key in state_data.keys():
                    if neuron_key in action_data and feature in state_data[neuron_key]:
                        feature_values.append(state_data[neuron_key][feature])
                        action_values.append(action_data[neuron_key][action])
                
                # Compute correlation if we have enough data
                if len(feature_values) >= 10:  # Minimum sample size
                    for method in self.correlation_methods:
                        if method == 'pearson':
                            corr, p_value = pearsonr(feature_values, action_values)
                        elif method == 'spearman':
                            corr, p_value = spearmanr(feature_values, action_values)
                        
                        correlations.append({
                            'feature': feature,
                            'action': action,
                            'correlation_method': method,
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(feature_values),
                            'significant': p_value < 0.05
                        })
        
        return correlations
    
    def plot_correlation_heatmap(self, results_df: pd.DataFrame, 
                               correlation_method: str = 'pearson',
                               significance_threshold: float = 0.05):
        """Create heatmap of significant correlations"""
        
        # Filter for significant correlations
        significant_results = results_df[
            (results_df['correlation_method'] == correlation_method) & 
            (results_df['p_value'] < significance_threshold)
        ]
        
        # Pivot to create heatmap data
        heatmap_data = significant_results.pivot_table(
            values='correlation', 
            index='feature', 
            columns='action',
            aggfunc='mean'  # Average across models/datasets
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 20))
        sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'Significant State-Action Correlations ({correlation_method.title()})')
        plt.tight_layout()
        plt.show()
        
        return heatmap_data



# Example usage
def run_comprehensive_analysis():
    """Example of how to run the full analysis"""
    
    # Create diverse models for testing
    models = [
        nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1)),
        nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 1)),
        # Add more diverse architectures
    ]
    
    # Create diverse datasets (regression tasks with different properties)
    datasets = get_all_datasets()
    
    # Run analysis
    analyzer = CorrelationAnalyzer(models, datasets)
    results = analyzer.run_full_analysis()
    
    # Display key findings
    print("Top 10 Strongest Correlations:")
    top_correlations = results.nlargest(10, 'correlation')
    print(top_correlations[['feature', 'action', 'correlation', 'p_value']])
    
    # Create visualization
    analyzer.plot_correlation_heatmap(results)
    
    return results

if __name__ == "__main__":
    # Run the analysis
    results = run_comprehensive_analysis()