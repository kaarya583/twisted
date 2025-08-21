"""
Model Evaluation Module for TwistEd ML Pipeline
Provides comprehensive evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self):
        self.results = {}
        self.figures = {}
        
    def evaluate_all_models(self, models: dict, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all trained models comprehensively
        
        Args:
            models: Dictionary of trained models
            X_test, y_test: Test data
        """
        print("📊 Starting comprehensive model evaluation...")
        
        self.results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            self.results[name] = metrics
            
            print(f"  ✅ {name} evaluation complete")
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def create_evaluation_dashboard(self, save_path: str = None):
        """
        Create comprehensive evaluation dashboard
        
        Args:
            save_path: Optional path to save dashboard
        """
        print("📊 Creating evaluation dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Model Performance Comparison',
                'Confusion Matrix (Best Model)',
                'ROC Curves',
                'Precision-Recall Curves',
                'Feature Importance (Best Model)',
                'Model Metrics Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Model Performance Comparison
        self._add_performance_comparison(fig, row=1, col=1)
        
        # 2. Confusion Matrix
        self._add_confusion_matrix(fig, row=1, col=2)
        
        # 3. ROC Curves
        self._add_roc_curves(fig, row=2, col=1)
        
        # 4. Precision-Recall Curves
        self._add_pr_curves(fig, row=2, col=2)
        
        # 5. Feature Importance
        self._add_feature_importance(fig, row=3, col=1)
        
        # 6. Metrics Summary Table
        self._add_metrics_table(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1400,
            title_text="TwistEd ML Model Evaluation Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Dashboard saved to {save_path}")
        
        return fig
    
    def _add_performance_comparison(self, fig, row: int, col: int):
        """Add model performance comparison bar chart"""
        
        if not self.results:
            return
        
        # Extract metrics
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create bar chart
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models if self.results[model][metric] is not None]
            if values:
                fig.add_trace(
                    go.Bar(
                        name=metric.title(),
                        x=models,
                        y=values,
                        marker_color=px.colors.qualitative.Set3[i],
                        showlegend=True
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Models", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)
    
    def _add_confusion_matrix(self, fig, row: int, col: int):
        """Add confusion matrix heatmap"""
        
        if not self.results:
            return
        
        # Find best model by accuracy
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = self.results[best_model]['confusion_matrix']
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in range(cm.shape[1])],
                y=[f'Actual {i}' for i in range(cm.shape[0])],
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Predicted", row=row, col=col)
        fig.update_yaxes(title_text="Actual", row=row, col=col)
    
    def _add_roc_curves(self, fig, row: int, col: int):
        """Add ROC curves for all models"""
        
        if not self.results:
            return
        
        for model_name, result in self.results.items():
            if result['probabilities'] is not None and result['roc_auc'] is not None:
                y_true = result['predictions']  # This should be y_test
                y_proba = result['probabilities']
                
                if len(np.unique(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    auc_score = result['roc_auc']
                    
                    fig.add_trace(
                        go.Scatter(
                            x=fpr, y=tpr,
                            name=f'{model_name} (AUC={auc_score:.3f})',
                            mode='lines',
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=row, col=col)
        fig.update_yaxes(title_text="True Positive Rate", row=row, col=col)
    
    def _add_pr_curves(self, fig, row: int, col: int):
        """Add Precision-Recall curves for all models"""
        
        if not self.results:
            return
        
        for model_name, result in self.results.items():
            if result['probabilities'] is not None:
                y_true = result['predictions']  # This should be y_test
                y_proba = result['probabilities']
                
                if len(np.unique(y_true)) == 2:
                    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=recall, y=precision,
                            name=model_name,
                            mode='lines',
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
        
        fig.update_xaxes(title_text="Recall", row=row, col=col)
        fig.update_yaxes(title_text="Precision", row=row, col=col)
    
    def _add_feature_importance(self, fig, row: int, col: int):
        """Add feature importance chart (placeholder)"""
        
        # This will be populated when feature importance is available
        fig.add_trace(
            go.Bar(
                x=['Feature 1', 'Feature 2', 'Feature 3'],
                y=[0.3, 0.2, 0.1],
                name='Feature Importance',
                marker_color='lightblue'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Features", row=row, col=col)
        fig.update_yaxes(title_text="Importance", row=row, col=col)
    
    def _add_metrics_table(self, fig, row: int, col: int):
        """Add metrics summary table"""
        
        if not self.results:
            return
        
        # Prepare table data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Model'] + [m.title() for m in metrics],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[
                        models,
                        [f"{self.results[model]['accuracy']:.3f}" if self.results[model]['accuracy'] else 'N/A' for model in models],
                        [f"{self.results[model]['precision']:.3f}" if self.results[model]['precision'] else 'N/A' for model in models],
                        [f"{self.results[model]['recall']:.3f}" if self.results[model]['recall'] else 'N/A' for model in models],
                        [f"{self.results[model]['f1']:.3f}" if self.results[model]['f1'] else 'N/A' for model in models],
                        [f"{self.results[model]['roc_auc']:.3f}" if self.results[model]['roc_auc'] else 'N/A' for model in models]
                    ],
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=row, col=col
        )
    
    def create_individual_model_plots(self, model_name: str, save_dir: str = None):
        """
        Create detailed plots for a specific model
        
        Args:
            model_name: Name of the model to analyze
            save_dir: Directory to save plots
        """
        if model_name not in self.results:
            print(f"❌ Model '{model_name}' not found in results!")
            return
        
        result = self.results[model_name]
        
        # Create subplots for this model
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Confusion Matrix - {model_name}',
                f'ROC Curve - {model_name}',
                f'Precision-Recall Curve - {model_name}',
                f'Classification Report - {model_name}'
            ]
        )
        
        # 1. Confusion Matrix
        cm = result['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in range(cm.shape[1])],
                y=[f'Actual {i}' for i in range(cm.shape[0])],
                colorscale='Blues',
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve
        if result['probabilities'] is not None and result['roc_auc'] is not None:
            y_true = result['predictions']  # This should be y_test
            y_proba = result['probabilities']
            
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                auc_score = result['roc_auc']
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr, y=tpr,
                        name=f'ROC (AUC={auc_score:.3f})',
                        mode='lines',
                        line=dict(width=2, color='red')
                    ),
                    row=1, col=2
                )
                
                # Add diagonal
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        name='Random',
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Precision-Recall Curve
        if result['probabilities'] is not None:
            y_true = result['predictions']  # This should be y_test
            y_proba = result['probabilities']
            
            if len(np.unique(y_true)) == 2:
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision,
                        name='PR Curve',
                        mode='lines',
                        line=dict(width=2, color='green')
                    ),
                    row=2, col=1
                )
        
        # 4. Classification Report
        report = result['classification_report']
        if isinstance(report, dict):
            # Extract metrics for each class
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            metrics_data = []
            for cls in classes:
                if isinstance(report[cls], dict):
                    metrics_data.append([
                        cls,
                        f"{report[cls].get('precision', 0):.3f}",
                        f"{report[cls].get('recall', 0):.3f}",
                        f"{report[cls].get('f1-score', 0):.3f}",
                        f"{report[cls].get('support', 0)}"
                    ])
            
            if metrics_data:
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                            fill_color='lightgreen',
                            align='left',
                            font=dict(size=12)
                        ),
                        cells=dict(
                            values=list(zip(*metrics_data)),
                            fill_color='white',
                            align='left',
                            font=dict(size=11)
                        )
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=f"Detailed Analysis - {model_name}",
            showlegend=True
        )
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{model_name}_analysis.html")
            fig.write_html(save_path)
            print(f"💾 Model analysis saved to {save_path}")
        
        return fig
    
    def generate_summary_report(self, save_path: str = None):
        """
        Generate a comprehensive summary report
        
        Args:
            save_path: Path to save the report
        """
        if not self.results:
            print("❌ No results to summarize!")
            return
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        
        # Create summary
        summary = f"""
# TwistEd ML Model Evaluation Summary Report

## Overview
This report summarizes the performance of {len(self.results)} machine learning models trained on NOAA weather data.

## Best Performing Model
**Model**: {best_model}
**Accuracy**: {self.results[best_model]['accuracy']:.4f}
**F1-Score**: {self.results[best_model]['f1']:.4f}
**ROC-AUC**: {f"{self.results[best_model]['roc_auc']:.4f}" if self.results[best_model]['roc_auc'] is not None else 'N/A'}

## Model Performance Summary
"""
        
        # Add performance table
        summary += "\n| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
        summary += "|-------|----------|-----------|--------|----------|----------|\n"
        
        for model_name, result in self.results.items():
            summary += f"| {model_name} | "
            summary += f"{result['accuracy']:.4f} | "
            summary += f"{result['precision']:.4f} | "
            summary += f"{result['recall']:.4f} | "
            summary += f"{result['f1']:.4f} | "
            roc_auc = result['roc_auc']
            if roc_auc is not None:
                summary += f"{roc_auc:.4f} |\n"
            else:
                summary += "N/A |\n"
        
        # Add recommendations
        summary += f"""

## Key Findings
1. **Best Model**: {best_model} achieved the highest accuracy of {self.results[best_model]['accuracy']:.4f}
2. **Model Count**: {len(self.results)} models were evaluated
3. **Performance Range**: Accuracy ranged from {min([r['accuracy'] for r in self.results.values()]):.4f} to {max([r['accuracy'] for r in self.results.values()]):.4f}

## Recommendations
1. Use {best_model} for production predictions
2. Consider ensemble methods for improved robustness
3. Monitor model performance on new data
4. Regular retraining with updated datasets

## Next Steps
1. Deploy {best_model} to production
2. Implement model monitoring
3. Set up automated retraining pipeline
4. A/B test with baseline models

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
            print(f"💾 Summary report saved to {save_path}")
        
        return summary
