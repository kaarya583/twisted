"""
Model Explainability Module for TwistEd ML Pipeline
Provides SHAP and LIME explanations for model predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Model explainability using SHAP and LIME
    Provides feature importance and prediction explanations
    """
    
    def __init__(self):
        self.shap_values = {}
        self.feature_importance = {}
        self.explainer = None
        
    def explain_model_shap(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          model_name: str = "model"):
        """
        Generate SHAP explanations for a model
        
        Args:
            model: Trained ML model
            X_train: Training features
            X_test: Test features
            model_name: Name identifier for the model
        """
        print(f"🔍 Generating SHAP explanations for {model_name}...")
        
        try:
            import shap
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For models with predict_proba
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_train)
            else:
                # For models without predict_proba
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_train)
            
            # Calculate SHAP values
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # Linear models
                shap_values = explainer.shap_values(X_test)
            
            # Store results
            self.shap_values[model_name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_names': X_test.columns.tolist()
            }
            
            print(f"  ✅ SHAP explanations generated for {model_name}")
            return explainer, shap_values
            
        except ImportError:
            print("  ⚠️ SHAP not available. Install with: pip install shap")
            return None, None
        except Exception as e:
            print(f"  ❌ SHAP explanation failed: {e}")
            return None, None
    
    def explain_model_lime(self, model, X_test: pd.DataFrame, sample_idx: int = 0, 
                          model_name: str = "model"):
        """
        Generate LIME explanations for a specific prediction
        
        Args:
            model: Trained ML model
            X_test: Test features
            sample_idx: Index of sample to explain
            model_name: Name identifier for the model
        """
        print(f"🔍 Generating LIME explanation for {model_name} sample {sample_idx}...")
        
        try:
            import lime
            import lime.lime_tabular
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test.values,
                feature_names=X_test.columns.tolist(),
                class_names=['0', '1'],  # Adjust based on your classes
                mode='classification'
            )
            
            # Explain the specific sample
            exp = explainer.explain_instance(
                X_test.iloc[sample_idx].values,
                model.predict_proba,
                num_features=min(10, len(X_test.columns))
            )
            
            print(f"  ✅ LIME explanation generated for {model_name}")
            return explainer, exp
            
        except ImportError:
            print("  ⚠️ LIME not available. Install with: pip install lime")
            return None, None
        except Exception as e:
            print(f"  ❌ LIME explanation failed: {e}")
            return None, None
    
    def create_shap_summary_plot(self, model_name: str, save_path: str = None):
        """
        Create SHAP summary plot
        
        Args:
            model_name: Name of the model to explain
            save_path: Optional path to save plot
        """
        if model_name not in self.shap_values:
            print(f"❌ No SHAP values found for {model_name}!")
            return None
        
        try:
            import shap
            
            shap_data = self.shap_values[model_name]
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            
            # Create summary plot
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {model_name}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 SHAP summary plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Failed to create SHAP summary plot: {e}")
            return None
    
    def create_shap_waterfall_plot(self, model_name: str, sample_idx: int = 0, 
                                  save_path: str = None):
        """
        Create SHAP waterfall plot for a specific prediction
        
        Args:
            model_name: Name of the model to explain
            sample_idx: Index of sample to explain
            save_path: Optional path to save plot
        """
        if model_name not in self.shap_values:
            print(f"❌ No SHAP values found for {model_name}!")
            return None
        
        try:
            import shap
            
            shap_data = self.shap_values[model_name]
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            
            # Create waterfall plot
            fig = plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=0,  # Adjust based on your model
                    data=None,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.title(f"SHAP Waterfall Plot - {model_name} Sample {sample_idx}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 SHAP waterfall plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Failed to create SHAP waterfall plot: {e}")
            return None
    
    def create_feature_importance_plot(self, model_name: str, save_path: str = None):
        """
        Create feature importance plot using SHAP values
        
        Args:
            model_name: Name of the model to explain
            save_path: Optional path to save plot
        """
        if model_name not in self.shap_values:
            print(f"❌ No SHAP values found for {model_name}!")
            return None
        
        try:
            shap_data = self.shap_values[model_name]
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)
            
            # Create horizontal bar plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=importance_df['feature'],
                x=importance_df['importance'],
                orientation='h',
                marker_color='lightblue',
                text=[f'{val:.4f}' for val in importance_df['importance']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Feature Importance (SHAP) - {model_name}",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=max(400, len(feature_names) * 20),
                width=800,
                showlegend=False
            )
            
            if save_path:
                fig.write_html(save_path)
                print(f"💾 Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Failed to create feature importance plot: {e}")
            return None
    
    def create_interactive_explanation_dashboard(self, model_name: str, X_test: pd.DataFrame, 
                                               save_path: str = None):
        """
        Create interactive explanation dashboard
        
        Args:
            model_name: Name of the model to explain
            X_test: Test features
            save_path: Optional path to save dashboard
        """
        if model_name not in self.shap_values:
            print(f"❌ No SHAP values found for {model_name}!")
            return None
        
        try:
            shap_data = self.shap_values[model_name]
            shap_values = shap_data['shap_values']
            feature_names = shap_data['feature_names']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f'SHAP Summary - {model_name}',
                    f'Feature Importance - {model_name}',
                    f'Sample Explanation - {model_name}',
                    f'SHAP Dependence - {model_name}'
                ]
            )
            
            # 1. SHAP Summary (simplified)
            mean_shap = np.abs(shap_values).mean(axis=0)
            top_features = np.argsort(mean_shap)[-10:]  # Top 10 features
            
            fig.add_trace(
                go.Bar(
                    x=[feature_names[i] for i in top_features],
                    y=[mean_shap[i] for i in top_features],
                    name='Mean |SHAP|',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # 2. Feature Importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True).tail(15)
            
            fig.add_trace(
                go.Bar(
                    y=importance_df['feature'],
                    x=importance_df['importance'],
                    orientation='h',
                    name='Feature Importance',
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            # 3. Sample Explanation (first sample)
            sample_idx = 0
            sample_shap = shap_values[sample_idx]
            top_sample_features = np.argsort(np.abs(sample_shap))[-10:]
            
            fig.add_trace(
                go.Bar(
                    x=[feature_names[i] for i in top_sample_features],
                    y=[sample_shap[i] for i in top_sample_features],
                    name=f'Sample {sample_idx} SHAP',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            # 4. SHAP Dependence (most important feature)
            if len(feature_names) > 0:
                most_important_idx = np.argmax(mean_shap)
                most_important_feature = feature_names[most_important_idx]
                
                # Get feature values and corresponding SHAP values
                feature_values = X_test.iloc[:, most_important_idx]
                feature_shap = shap_values[:, most_important_idx]
                
                fig.add_trace(
                    go.Scatter(
                        x=feature_values,
                        y=feature_shap,
                        mode='markers',
                        name=f'{most_important_feature} vs SHAP',
                        marker=dict(color='red', size=5, opacity=0.6)
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text=f"Model Explanation Dashboard - {model_name}",
                showlegend=True
            )
            
            if save_path:
                fig.write_html(save_path)
                print(f"💾 Explanation dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"❌ Failed to create explanation dashboard: {e}")
            return None
    
    def explain_prediction(self, model, X_sample: pd.DataFrame, model_name: str = "model"):
        """
        Explain a specific prediction
        
        Args:
            model: Trained ML model
            X_sample: Single sample to explain
            model_name: Name identifier for the model
        """
        print(f"🔍 Explaining prediction for {model_name}...")
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict(X_sample)[0]
            probability = model.predict_proba(X_sample)[0]
        else:
            prediction = model.predict(X_sample)[0]
            probability = None
        
        print(f"  📊 Prediction: {prediction}")
        if probability is not None:
            print(f"  📈 Probabilities: {probability}")
        
        # Generate SHAP explanation if available
        if model_name in self.shap_values:
            try:
                import shap
                
                shap_data = self.shap_values[model_name]
                explainer = shap_data['explainer']
                
                # Get SHAP values for this sample
                if hasattr(explainer, 'shap_values'):
                    sample_shap = explainer.shap_values(X_sample)
                    if isinstance(sample_shap, list):
                        sample_shap = sample_shap[1] if len(sample_shap) > 1 else sample_shap[0]
                    
                    # Get top contributing features
                    feature_contributions = list(zip(
                        shap_data['feature_names'],
                        sample_shap[0]
                    ))
                    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print(f"  🔍 Top contributing features:")
                    for feature, contribution in feature_contributions[:5]:
                        print(f"    {feature}: {contribution:.4f}")
                
            except Exception as e:
                print(f"    ⚠️ SHAP explanation failed: {e}")
        
        return {
            'prediction': prediction,
            'probability': probability,
            'feature_contributions': feature_contributions if 'feature_contributions' in locals() else None
        }
    
    def create_comparative_explanation(self, models: dict, X_test: pd.DataFrame, 
                                      save_path: str = None):
        """
        Create comparative explanation across multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            save_path: Optional path to save comparison
        """
        print("🔍 Creating comparative explanation across models...")
        
        # Get feature importance for all models
        model_importance = {}
        
        for name, model in models.items():
            if name in self.shap_values:
                shap_data = self.shap_values[name]
                shap_values = shap_data['shap_values']
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                # Get top 10 features
                top_indices = np.argsort(mean_shap)[-10:]
                model_importance[name] = {
                    'features': [shap_data['feature_names'][i] for i in top_indices],
                    'importance': [mean_shap[i] for i in top_indices]
                }
        
        if not model_importance:
            print("❌ No SHAP values available for comparison!")
            return None
        
        # Create comparison plot
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, (model_name, importance_data) in enumerate(model_importance.items()):
            fig.add_trace(go.Bar(
                name=model_name,
                x=importance_data['features'],
                y=importance_data['importance'],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Feature Importance Comparison Across Models",
            xaxis_title="Features",
            yaxis_title="Mean |SHAP Value|",
            barmode='group',
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Comparative explanation saved to {save_path}")
        
        return fig
    
    def save_explanations(self, save_dir: str):
        """
        Save all explanation data
        
        Args:
            save_dir: Directory to save explanations
        """
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save SHAP values summary
        for model_name, shap_data in self.shap_values.items():
            # Calculate summary statistics
            shap_values = shap_data['shap_values']
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            summary = {
                'model_name': model_name,
                'feature_names': shap_data['feature_names'],
                'mean_importance': mean_shap.tolist(),
                'total_samples': len(shap_values),
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            # Save summary
            summary_path = os.path.join(save_dir, f"{model_name}_shap_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"💾 Saved SHAP summary for {model_name}")
        
        print(f"💾 All explanations saved to {save_dir}")
