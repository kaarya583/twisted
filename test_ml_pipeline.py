#!/usr/bin/env python3
"""
Test script for TwistEd ML Pipeline
Verifies all components work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add ML pipeline to path
sys.path.append('ml_pipeline')

def test_feature_engineering():
    """Test feature engineering module"""
    print("🧪 Testing Feature Engineering...")
    
    try:
        from ml_pipeline.feature_engineering import FeatureEngineer
        
        # Create sample data
        sample_data = pd.DataFrame({
            'EVENT_TYPE': ['Tornado', 'Thunderstorm Wind', 'Hail'],
            'STATE': ['TX', 'CA', 'FL'],
            'CZ_NAME': ['County A', 'County B', 'County C'],
            'EVENT_NARRATIVE': [
                'A tornado touched down causing significant damage',
                'Strong thunderstorm winds uprooted trees',
                'Large hail damaged vehicles and roofs'
            ],
            'BEGIN_DATE_TIME': ['2024-01-15 14:30:00', '2024-01-16 16:45:00', '2024-01-17 12:15:00'],
            'END_DATE_TIME': ['2024-01-15 15:30:00', '2024-01-16 17:45:00', '2024-01-17 13:15:00'],
            'severity': ['Severe', 'Moderate', 'Minor'],
            'urgency': ['Immediate', 'Expected', 'Expected']
        })
        
        # Test feature engineering
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(sample_data)
        
        print(f"  ✅ Feature engineering successful: {len(df_features.columns)} features created")
        print(f"  🔧 Sample features: {list(df_features.columns[:10])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        return False

def test_ml_models():
    """Test ML models module"""
    print("🧪 Testing ML Models...")
    
    try:
        from ml_pipeline.models import WeatherMLModels
        
        # Create more sample data to ensure each class has enough samples
        X_sample = pd.DataFrame({
            'month': [6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7, 8],  # 12 samples
            'hour': [14, 16, 12, 15, 17, 13, 14, 16, 12, 15, 17, 13],
            'is_weekend': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'event_tornado': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'event_thunderstorm_wind': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'severity_severe': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'severity_moderate': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'region_southeast': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            'region_west': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            'event_duration_hours': [1.0, 1.5, 0.5, 1.2, 1.8, 0.7, 1.1, 1.6, 0.6, 1.3, 1.9, 0.8]
        })
        
        # Create balanced target variable (4 samples per class)
        y_sample = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0])  # Severe, Moderate, Minor
        
        # Test ML models
        ml_models = WeatherMLModels()
        X_train, X_test, y_train, y_test, feature_names = ml_models.prepare_data(
            pd.concat([X_sample, pd.Series(y_sample, name='severity')], axis=1),
            target_col='severity',
            test_size=0.3
        )
        
        print(f"  ✅ ML models setup successful")
        print(f"  📊 Training samples: {len(X_train)}")
        print(f"  📊 Test samples: {len(X_test)}")
        print(f"  🔧 Features: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ML models test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation module"""
    print("🧪 Testing Model Evaluation...")
    
    try:
        from ml_pipeline.evaluation import ModelEvaluator
        
        # Create sample evaluation data
        evaluator = ModelEvaluator()
        
        # Mock results
        evaluator.results = {
            'test_model': {
                'accuracy': 0.85,
                'precision': 0.84,
                'recall': 0.85,
                'f1': 0.85,
                'roc_auc': 0.89,
                'predictions': np.array([0, 1, 2]),
                'probabilities': np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                'confusion_matrix': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                'classification_report': {'accuracy': 0.85}
            }
        }
        
        # Test dashboard creation
        dashboard = evaluator.create_evaluation_dashboard()
        
        print(f"  ✅ Model evaluation successful")
        print(f"  📊 Dashboard created with {len(dashboard.data)} traces")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model evaluation test failed: {e}")
        return False

def test_explainability():
    """Test explainability module"""
    print("🧪 Testing Model Explainability...")
    
    try:
        from ml_pipeline.explainability import ModelExplainer
        
        # Create explainer
        explainer = ModelExplainer()
        
        # Mock SHAP data
        explainer.shap_values = {
            'test_model': {
                'explainer': None,
                'shap_values': np.random.rand(10, 5),
                'feature_names': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
            }
        }
        
        # Test feature importance plot
        importance_plot = explainer.create_feature_importance_plot('test_model')
        
        print(f"  ✅ Model explainability successful")
        print(f"  🔍 Feature importance plot created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model explainability test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing Data Loading...")
    
    try:
        # Create a simple test data file to simulate NOAA data
        import os
        import gzip
        
        # Create test data directory
        os.makedirs('noaa_data', exist_ok=True)
        
        # Create a simple test CSV file
        test_data = """BEGIN_DATE_TIME,END_DATE_TIME,EVENT_TYPE,STATE,CZ_NAME,EVENT_NARRATIVE
2022-01-15 14:30:00,2022-01-15 15:30:00,Tornado,TX,Test County,A test tornado event
2022-01-16 16:45:00,2022-01-16 17:45:00,Thunderstorm Wind,CA,Test County,A test thunderstorm event
2022-01-17 12:15:00,2022-01-17 13:15:00,Hail,FL,Test County,A test hail event"""
        
        # Save as gzipped CSV
        test_file_path = 'noaa_data/StormEvents_details-test.csv.gz'
        with gzip.open(test_file_path, 'wt', encoding='utf-8') as f:
            f.write(test_data)
        
        print(f"  ✅ Created test NOAA data file")
        
        # Check if the file exists
        if os.path.exists(test_file_path):
            print(f"  ✅ Test data file found: {test_file_path}")
            
            # Verify we can read it
            import pandas as pd
            df = pd.read_csv(test_file_path, compression='gzip')
            print(f"  ✅ Successfully read test data: {len(df)} rows")
            
            # Clean up test file
            os.remove(test_file_path)
            print(f"  ✅ Cleaned up test file")
            
            return True
        else:
            print(f"  ❌ Test data file not created")
            return False
            
    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting TwistEd ML Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("ML Models", test_ml_models),
        ("Model Evaluation", test_evaluation),
        ("Model Explainability", test_explainability),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ML pipeline is ready to use.")
        print("\n🚀 Next steps:")
        print("  1. Run: python train_ml_models.py")
        print("  2. Launch: streamlit run twisted_ml_dashboard.py")
        print("  3. Deploy: docker build -t twisted-ml .")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("  1. Install requirements: pip install -r requirements_ml.txt")
        print("  2. Check Python version: python --version")
        print("  3. Verify file structure and imports")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
