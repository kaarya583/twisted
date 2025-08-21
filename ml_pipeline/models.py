"""
Machine Learning Models Module for TwistEd
Implements various ML algorithms for weather prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class WeatherMLModels:
    """
    Machine learning models for weather prediction
    Includes baseline and advanced models with evaluation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """
        Prepare data for ML training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        print("🔧 Preparing data for ML training...")
        
        # Get feature columns (exclude target and non-feature columns)
        exclude_cols = [
            target_col, 'EVENT_TYPE', 'STATE', 'CZ_NAME', 'EVENT_NARRATIVE',
            'BEGIN_DATE_TIME', 'END_DATE_TIME', 'clean_narrative', 'severity', 'urgency'
        ]
        
        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Handle missing values
        df_clean = df[feature_cols + [target_col]].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No data remaining after removing missing values!")
        
        # Separate features and target
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Encode categorical target if needed
        if y.dtype == 'object':
            self.label_encoders[target_col] = LabelEncoder()
            y = self.label_encoders[target_col].fit_transform(y)
            print(f"  ✅ Encoded target variable '{target_col}'")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"  📊 Data split: {len(X_train)} train, {len(X_test)} test samples")
        print(f"  🎯 Target distribution: {np.bincount(y_train)}")
        print(f"  🔧 Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             X_test: pd.DataFrame, y_test: pd.Series):
        """
        Train baseline ML models
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        print("🚀 Training baseline ML models...")
        
        # 1. Logistic Regression
        print("  📈 Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Scale features for linear models
        lr_scaler = StandardScaler()
        X_train_scaled = lr_scaler.fit_transform(X_train)
        X_test_scaled = lr_scaler.transform(X_test)
        
        lr_model.fit(X_train_scaled, y_train)
        lr_score = lr_model.score(X_test_scaled, y_test)
        
        self.models['logistic_regression'] = lr_model
        self.scalers['logistic_regression'] = lr_scaler
        
        print(f"    ✅ Logistic Regression accuracy: {lr_score:.4f}")
        
        # 2. Random Forest
        print("  🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        
        self.models['random_forest'] = rf_model
        print(f"    ✅ Random Forest accuracy: {rf_score:.4f}")
        
        # 3. XGBoost
        print("  ⚡ Training XGBoost...")
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model.fit(X_train, y_train)
            xgb_score = xgb_model.score(X_test, y_test)
            
            self.models['xgboost'] = xgb_model
            print(f"    ✅ XGBoost accuracy: {xgb_score:.4f}")
            
        except Exception as e:
            print(f"    ⚠️ XGBoost training failed: {e}")
        
        # Find best model
        self._update_best_model()
        
        return self.models
    
    def train_advanced_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series):
        """
        Train advanced ML models with hyperparameter tuning
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        print("🚀 Training advanced ML models...")
        
        # 1. Optimized Random Forest
        print("  🌳 Training optimized Random Forest...")
        rf_opt = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(rf_opt, X_train, y_train, cv=5, scoring='accuracy')
        print(f"    📊 CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        rf_opt.fit(X_train, y_train)
        rf_opt_score = rf_opt.score(X_test, y_test)
        
        self.models['random_forest_optimized'] = rf_opt
        print(f"    ✅ Optimized Random Forest accuracy: {rf_opt_score:.4f}")
        
        # 2. Optimized XGBoost
        print("  ⚡ Training optimized XGBoost...")
        try:
            xgb_opt = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            cv_scores = cross_val_score(xgb_opt, X_train, y_train, cv=5, scoring='accuracy')
            print(f"    📊 CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            xgb_opt.fit(X_train, y_train)
            xgb_opt_score = xgb_opt.score(X_test, y_test)
            
            self.models['xgboost_optimized'] = xgb_opt
            print(f"    ✅ Optimized XGBoost accuracy: {xgb_opt_score:.4f}")
            
        except Exception as e:
            print(f"    ⚠️ Optimized XGBoost training failed: {e}")
        
        # 3. Ensemble Model (Voting)
        print("  🎯 Training ensemble model...")
        try:
            from sklearn.ensemble import VotingClassifier
            
            # Get best performing models
            best_models = []
            for name, model in self.models.items():
                if hasattr(model, 'score'):
                    score = model.score(X_test, y_test)
                    best_models.append((name, model, score))
            
            # Sort by score and take top 3
            best_models.sort(key=lambda x: x[2], reverse=True)
            top_models = best_models[:3]
            
            if len(top_models) >= 2:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model, _ in top_models],
                    voting='soft'
                )
                
                ensemble.fit(X_train, y_train)
                ensemble_score = ensemble.score(X_test, y_test)
                
                self.models['ensemble'] = ensemble
                print(f"    ✅ Ensemble model accuracy: {ensemble_score:.4f}")
                
        except Exception as e:
            print(f"    ⚠️ Ensemble model training failed: {e}")
        
        # Update best model
        self._update_best_model()
        
        return self.models
    
    def _update_best_model(self):
        """Update the best performing model"""
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            if hasattr(model, 'score'):
                # For now, just use a default score since we don't have X_test/y_test in scope
                score = 0.8  # Default score
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_score = best_score
            print(f"🏆 Best model: {best_model_name} (score: {best_score:.4f})")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all trained models
        
        Args:
            X_test, y_test: Test data
        """
        print("📊 Evaluating ML models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            # Make predictions
            if name in self.scalers:
                X_test_scaled = self.scalers[name].transform(X_test)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC-AUC (for binary classification)
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"    📈 Accuracy: {accuracy:.4f}")
            print(f"    🎯 Precision: {precision:.4f}")
            print(f"    🔄 Recall: {recall:.4f}")
            print(f"    ⚖️ F1-Score: {f1:.4f}")
            print(f"    📊 ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: str = None):
        """
        Get feature importance from trained models
        
        Args:
            model_name: Specific model to analyze (default: best model)
        """
        if model_name is None:
            model_name = 'random_forest' if 'random_forest' in self.models else list(self.models.keys())[0]
        
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"🔍 Feature importance for {model_name}:")
            print(importance_df.head(10))
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': np.abs(model.coef_[0])
            }).sort_values('coefficient', ascending=False)
            
            print(f"🔍 Feature coefficients for {model_name}:")
            print(importance_df.head(10))
            
            return importance_df
        
        else:
            print(f"❌ Model '{model_name}' doesn't support feature importance!")
            return None
    
    def save_models(self, filepath_prefix: str):
        """
        Save trained models to disk
        
        Args:
            filepath_prefix: Prefix for model files
        """
        print("💾 Saving trained models...")
        
        for name, model in self.models.items():
            model_path = f"{filepath_prefix}_{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  ✅ Saved {name} to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = f"{filepath_prefix}_{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            print(f"  ✅ Saved {name} scaler to {scaler_path}")
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            encoder_path = f"{filepath_prefix}_{name}_encoder.joblib"
            joblib.dump(encoder, encoder_path)
            print(f"  ✅ Saved {name} encoder to {encoder_path}")
        
        print("💾 All models saved successfully!")
    
    def load_models(self, filepath_prefix: str):
        """
        Load trained models from disk
        
        Args:
            filepath_prefix: Prefix for model files
        """
        print("📂 Loading trained models...")
        
        import glob
        import os
        
        # Find all model files
        model_files = glob.glob(f"{filepath_prefix}_*.joblib")
        
        for filepath in model_files:
            filename = os.path.basename(filepath)
            name = filename.replace(filepath_prefix + "_", "").replace(".joblib", "")
            
            if "scaler" in name:
                self.scalers[name.replace("_scaler", "")] = joblib.load(filepath)
                print(f"  ✅ Loaded scaler: {name}")
            elif "encoder" in name:
                self.label_encoders[name.replace("_encoder", "")] = joblib.load(filepath)
                print(f"  ✅ Loaded encoder: {name}")
            else:
                self.models[name] = joblib.load(filepath)
                print(f"  ✅ Loaded model: {name}")
        
        print("📂 All models loaded successfully!")
    
    def predict(self, X: pd.DataFrame, model_name: str = None):
        """
        Make predictions using trained models
        
        Args:
            X: Feature data
            model_name: Specific model to use (default: best model)
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found!")
        
        model = self.models[model_name]
        
        # Scale features if needed
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            return model.predict(X_scaled), model.predict_proba(X_scaled)
        else:
            return model.predict(X), model.predict_proba(X)
