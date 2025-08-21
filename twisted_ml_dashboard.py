#!/usr/bin/env python3
"""
TwistEd ML-Enhanced Dashboard
Advanced weather dashboard with ML predictions and explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import joblib
import os
import sys
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Add ML pipeline to path
sys.path.append('ml_pipeline')

# Page configuration
st.set_page_config(
    page_title="🌪️ TwistEd ML Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .alert-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #f59e0b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #10b981;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ml_models_loaded' not in st.session_state:
    st.session_state.ml_models_loaded = False
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None

@st.cache_data
def load_ml_models():
    """Load trained ML models"""
    try:
        models = {}
        scalers = {}
        encoders = {}
        
        model_dir = 'ml_outputs/models'
        if os.path.exists(model_dir):
            # Load models
            for file in os.listdir(model_dir):
                if file.endswith('.joblib') and 'scaler' not in file and 'encoder' not in file:
                    model_name = file.replace('twisted_weather_', '').replace('.joblib', '')
                    model_path = os.path.join(model_dir, file)
                    models[model_name] = joblib.load(model_path)
                    st.success(f"✅ Loaded model: {model_name}")
                
                elif 'scaler' in file:
                    model_name = file.replace('twisted_weather_', '').replace('_scaler.joblib', '')
                    scaler_path = os.path.join(model_dir, file)
                    scalers[model_name] = joblib.load(scaler_path)
                
                elif 'encoder' in file:
                    model_name = file.replace('twisted_weather_', '').replace('_encoder.joblib', '')
                    encoder_path = os.path.join(model_dir, file)
                    encoders[model_name] = joblib.load(encoder_path)
        
        return models, scalers, encoders
    except Exception as e:
        st.error(f"❌ Failed to load ML models: {e}")
        return {}, {}, {}

@st.cache_data
def load_feature_engineering_config():
    """Load feature engineering configuration"""
    try:
        config_path = 'ml_outputs/features/feature_config.json'
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"❌ Failed to load feature config: {e}")
        return None

def create_ml_features(weather_data):
    """Create ML features from weather data"""
    try:
        # This is a simplified version - in production, use the full FeatureEngineer
        features = {}
        
        # Temporal features
        if 'sent' in weather_data:
            sent_time = pd.to_datetime(weather_data['sent'])
            features['hour'] = sent_time.hour
            features['day_of_week'] = sent_time.dayofweek
            features['is_weekend'] = int(sent_time.dayofweek in [5, 6])
            features['month'] = sent_time.month
        
        # Event type features
        event_type = weather_data.get('event', 'Unknown')
        features['event_tornado'] = int('tornado' in event_type.lower())
        features['event_thunderstorm'] = int('thunderstorm' in event_type.lower())
        features['event_flood'] = int('flood' in event_type.lower())
        features['event_hail'] = int('hail' in event_type.lower())
        
        # Severity features
        severity = weather_data.get('severity', 'Minor')
        features['severity_minor'] = int(severity == 'Minor')
        features['severity_moderate'] = int(severity == 'Moderate')
        features['severity_severe'] = int(severity == 'Severe')
        features['severity_extreme'] = int(severity == 'Extreme')
        
        # Geographic features (simplified)
        features['region_northeast'] = 0
        features['region_southeast'] = 0
        features['region_midwest'] = 0
        features['region_southwest'] = 0
        features['region_west'] = 0
        
        # Add default values for missing features
        default_features = {
            'year': 2024,
            'day_of_year': 1,
            'event_duration_hours': 1.0,
            'narrative_length': 100,
            'word_count': 20,
            'state_encoded': 0,
            'county_encoded': 0,
            'latitude': 39.5,
            'longitude': -98.35,
            'lat_bin': 5,
            'lon_bin': 5
        }
        
        for feature, default_value in default_features.items():
            if feature not in features:
                features[feature] = default_value
        
        return pd.DataFrame([features])
        
    except Exception as e:
        st.error(f"❌ Feature creation failed: {e}")
        return pd.DataFrame()

def make_ml_prediction(features_df, models, scalers, encoders):
    """Make ML predictions"""
    try:
        predictions = {}
        
        for model_name, model in models.items():
            try:
                # Scale features if scaler exists
                if model_name in scalers:
                    features_scaled = scalers[model_name].transform(features_df)
                    pred = model.predict(features_scaled)[0]
                    proba = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(features_df)[0]
                    proba = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': proba
                }
                
            except Exception as e:
                st.warning(f"⚠️ Prediction failed for {model_name}: {e}")
                continue
        
        return predictions
        
    except Exception as e:
        st.error(f"❌ ML prediction failed: {e}")
        return {}

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌪️ TwistEd ML-Enhanced Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Load ML models
    if not st.session_state.ml_models_loaded:
        with st.sidebar:
            st.info("🔄 Loading ML models...")
            models, scalers, encoders = load_ml_models()
            if models:
                st.session_state.ml_models_loaded = True
                st.session_state.models = models
                st.session_state.scalers = scalers
                st.session_state.encoders = encoders
                st.success(f"✅ Loaded {len(models)} ML models")
            else:
                st.warning("⚠️ No ML models found. Run training pipeline first.")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌦️ Live Weather & ML", 
        "🤖 ML Predictions", 
        "📊 Model Performance", 
        "🔍 Explainability", 
        "📈 Historical Analysis"
    ])
    
    with tab1:
        st.header("🌦️ Live Weather Data with ML Predictions")
        
        # Load current weather data
        try:
            from twisted import get_weather_context, API_URL, HEADERS
            
            # Get current weather context
            response = requests.get(API_URL, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                data = response.json()
                alerts = data.get("features", [])
                
                if alerts:
                    st.success(f"✅ Loaded {len(alerts)} active weather alerts")
                    
                    # Create ML features and predictions
                    if st.session_state.ml_models_loaded:
                        st.subheader("🤖 ML Severity Predictions")
                        
                        # Select an alert to analyze
                        alert_options = [f"{alert['properties']['event']} - {alert['properties']['areaDesc']}" 
                                       for alert in alerts[:10]]
                        selected_alert = st.selectbox("Select alert for ML analysis:", alert_options)
                        
                        if selected_alert:
                            # Find the selected alert
                            alert_idx = alert_options.index(selected_alert)
                            alert = alerts[alert_idx]
                            props = alert['properties']
                            
                            # Create features
                            features_df = create_ml_features(props)
                            
                            if not features_df.empty:
                                # Make predictions
                                predictions = make_ml_prediction(
                                    features_df, 
                                    st.session_state.models, 
                                    st.session_state.scalers, 
                                    st.session_state.encoders
                                )
                                
                                if predictions:
                                    st.session_state.current_predictions = predictions
                                    
                                    # Display predictions
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("### 📊 Model Predictions")
                                        for model_name, pred_data in predictions.items():
                                            pred = pred_data['prediction']
                                            proba = pred_data['probability']
                                            
                                            # Map prediction to severity
                                            severity_map = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
                                            predicted_severity = severity_map.get(pred, 'Unknown')
                                            
                                            st.markdown(f"""
                                            **{model_name.title()}**
                                            - Predicted Severity: {predicted_severity}
                                            - Confidence: {max(proba) if proba is not None else 'N/A':.3f}
                                            """)
                                    
                                    with col2:
                                        st.markdown("### 🎯 Alert Details")
                                        st.markdown(f"""
                                        **Event**: {props['event']}
                                        **Location**: {props['areaDesc']}
                                        **Current Severity**: {props.get('severity', 'Unknown')}
                                        **Urgency**: {props.get('urgency', 'Unknown')}
                                        **Headline**: {props.get('headline', 'No headline')}
                                        """)
                                    
                                    # Show prediction vs actual
                                    st.markdown("### 🔍 Prediction Analysis")
                                    actual_severity = props.get('severity', 'Unknown')
                                    
                                    # Create comparison chart
                                    model_names = list(predictions.keys())
                                    predicted_severities = []
                                    
                                    for model_name in model_names:
                                        pred = predictions[model_name]['prediction']
                                        predicted_severities.append(severity_map.get(pred, 'Unknown'))
                                    
                                    comparison_df = pd.DataFrame({
                                        'Model': model_names,
                                        'Predicted': predicted_severities,
                                        'Actual': [actual_severity] * len(model_names)
                                    })
                                    
                                    st.dataframe(comparison_df)
                                    
                                    # Accuracy indicator
                                    correct_predictions = sum(1 for pred, actual in zip(predicted_severities, [actual_severity] * len(model_names)) if pred == actual)
                                    accuracy = correct_predictions / len(model_names)
                                    
                                    if accuracy >= 0.8:
                                        st.success(f"🎯 High Prediction Accuracy: {accuracy:.1%}")
                                    elif accuracy >= 0.6:
                                        st.warning(f"⚠️ Moderate Prediction Accuracy: {accuracy:.1%}")
                                    else:
                                        st.error(f"❌ Low Prediction Accuracy: {accuracy:.1%}")
                                    
                                else:
                                    st.warning("⚠️ No predictions generated")
                            else:
                                st.warning("⚠️ Failed to create features")
                        else:
                            st.info("ℹ️ Select an alert to see ML predictions")
                    else:
                        st.info("ℹ️ ML models not loaded. Check sidebar for status.")
                    
                    # Display alerts in a table
                    st.subheader("📋 Active Weather Alerts")
                    alerts_data = []
                    for alert in alerts[:20]:  # Show first 20
                        props = alert['properties']
                        alerts_data.append({
                            'Event': props['event'],
                            'Location': props['areaDesc'],
                            'Severity': props.get('severity', 'Unknown'),
                            'Urgency': props.get('urgency', 'Unknown'),
                            'Sent': props.get('sent', 'Unknown'),
                            'Expires': props.get('expires', 'Unknown')
                        })
                    
                    alerts_df = pd.DataFrame(alerts_data)
                    st.dataframe(alerts_df, use_container_width=True)
                    
                else:
                    st.info("ℹ️ No active weather alerts")
                    
        except Exception as e:
            st.error(f"❌ Failed to load weather data: {e}")
    
    with tab2:
        st.header("🤖 ML Model Predictions")
        
        if not st.session_state.ml_models_loaded:
            st.warning("⚠️ ML models not loaded. Please run the training pipeline first.")
        else:
            st.success(f"✅ {len(st.session_state.models)} ML models available")
            
            # Model selection
            model_name = st.selectbox("Select ML model:", list(st.session_state.models.keys()))
            
            if model_name:
                model = st.session_state.models[model_name]
                
                # Model info
                st.markdown(f"### 📊 Model: {model_name}")
                
                # Show model type and parameters
                if hasattr(model, 'n_estimators'):
                    st.info(f"**Type**: Ensemble (Random Forest/XGBoost)")
                    st.info(f"**Estimators**: {model.n_estimators}")
                elif hasattr(model, 'C'):
                    st.info(f"**Type**: Linear (Logistic Regression)")
                    st.info(f"**Regularization**: C={model.C}")
                else:
                    st.info(f"**Type**: {type(model).__name__}")
                
                # Interactive prediction
                st.markdown("### 🎯 Make a Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Input features
                    st.markdown("#### Input Features")
                    
                    # Event type
                    event_type = st.selectbox("Event Type:", [
                        "Tornado", "Thunderstorm Wind", "Hail", "Flash Flood", 
                        "Flood", "Winter Storm", "High Wind", "Heavy Rain"
                    ])
                    
                    # Severity
                    severity = st.selectbox("Severity:", ["Minor", "Moderate", "Severe", "Extreme"])
                    
                    # Month
                    month = st.slider("Month:", 1, 12, 6)
                    
                    # Hour
                    hour = st.slider("Hour of Day:", 0, 23, 12)
                    
                    # Weekend
                    is_weekend = st.checkbox("Weekend")
                
                with col2:
                    # Additional features
                    st.markdown("#### Additional Features")
                    
                    # State
                    state = st.selectbox("State:", [
                        "TX", "CA", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"
                    ])
                    
                    # Duration
                    duration = st.slider("Event Duration (hours):", 0.1, 24.0, 1.0)
                    
                    # Narrative length
                    narrative_length = st.slider("Narrative Length:", 10, 500, 100)
                    
                    # Latitude/Longitude
                    lat = st.slider("Latitude:", 25.0, 50.0, 39.5)
                    lon = st.slider("Longitude:", -125.0, -65.0, -98.35)
                
                # Create features
                if st.button("🚀 Generate Prediction"):
                    # Create feature vector
                    features = {
                        'event_tornado': int(event_type == "Tornado"),
                        'event_thunderstorm_wind': int(event_type == "Thunderstorm Wind"),
                        'event_flood': int("Flood" in event_type),
                        'event_hail': int(event_type == "Hail"),
                        'severity_minor': int(severity == "Minor"),
                        'severity_moderate': int(severity == "Moderate"),
                        'severity_severe': int(severity == "Severe"),
                        'severity_extreme': int(severity == "Extreme"),
                        'month': month,
                        'hour': hour,
                        'is_weekend': int(is_weekend),
                        'event_duration_hours': duration,
                        'narrative_length': narrative_length,
                        'latitude': lat,
                        'longitude': lon,
                        'year': 2024,
                        'day_of_year': 1,
                        'day_of_week': 5 if is_weekend else 1,
                        'word_count': narrative_length // 5,
                        'state_encoded': 0,
                        'county_encoded': 0,
                        'lat_bin': int((lat - 25.0) / 2.5),
                        'lon_bin': int((lon + 125.0) / 6.0),
                        'is_spring': int(month in [3, 4, 5]),
                        'is_summer': int(month in [6, 7, 8]),
                        'is_fall': int(month in [9, 10, 11]),
                        'is_winter': int(month in [12, 1, 2])
                    }
                    
                    # Add region features
                    northeast_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA']
                    southeast_states = ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA']
                    midwest_states = ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS']
                    southwest_states = ['OK', 'TX', 'NM', 'AZ']
                    west_states = ['CO', 'WY', 'MT', 'ID', 'WA', 'OR', 'CA', 'NV', 'UT']
                    
                    features['region_northeast'] = int(state in northeast_states)
                    features['region_southeast'] = int(state in southeast_states)
                    features['region_midwest'] = int(state in midwest_states)
                    features['region_southwest'] = int(state in southwest_states)
                    features['region_west'] = int(state in west_states)
                    
                    features_df = pd.DataFrame([features])
                    
                    # Make prediction
                    try:
                        if model_name in st.session_state.scalers:
                            features_scaled = st.session_state.scalers[model_name].transform(features_df)
                            prediction = model.predict(features_scaled)[0]
                            probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
                        else:
                            prediction = model.predict(features_df)[0]
                            probability = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else None
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        
                        severity_map = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
                        predicted_severity = severity_map.get(prediction, 'Unknown')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Severity", predicted_severity)
                        
                        with col2:
                            if probability is not None:
                                confidence = max(probability)
                                st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            st.metric("Model", model_name.title())
                        
                        # Show probability distribution
                        if probability is not None:
                            st.markdown("#### 📈 Probability Distribution")
                            
                            prob_df = pd.DataFrame({
                                'Severity': ['Minor', 'Moderate', 'Severe'],
                                'Probability': probability
                            })
                            
                            fig = px.bar(prob_df, x='Severity', y='Probability', 
                                        title=f"Prediction Probabilities - {model_name.title()}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            st.markdown("#### 🔍 Top Contributing Features")
                            
                            # Get feature importance
                            if model_name in st.session_state.scalers:
                                # For scaled models, we need to handle this differently
                                st.info("Feature importance analysis available in Explainability tab")
                            else:
                                feature_importance = model.feature_importances_
                                feature_names = list(features.keys())
                                
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': feature_importance
                                }).sort_values('Importance', ascending=False)
                                
                                st.dataframe(importance_df.head(10))
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
    
    with tab3:
        st.header("📊 ML Model Performance")
        
        # Check if evaluation results exist
        eval_dir = 'ml_outputs/evaluations'
        if os.path.exists(eval_dir):
            st.success("✅ Model evaluation results found")
            
            # Load evaluation dashboard
            dashboard_path = os.path.join(eval_dir, 'evaluation_dashboard.html')
            if os.path.exists(dashboard_path):
                st.markdown("### 📈 Interactive Evaluation Dashboard")
                
                # Display the HTML dashboard
                with open(dashboard_path, 'r') as f:
                    dashboard_html = f.read()
                
                st.components.v1.html(dashboard_html, height=800)
            else:
                st.warning("⚠️ Evaluation dashboard not found")
            
            # Load summary report
            summary_path = os.path.join(eval_dir, 'summary_report.md')
            if os.path.exists(summary_path):
                st.markdown("### 📝 Summary Report")
                with open(summary_path, 'r') as f:
                    summary_content = f.read()
                st.markdown(summary_content)
        else:
            st.warning("⚠️ No evaluation results found. Run the training pipeline first.")
            st.info("To generate evaluation results, run: `python train_ml_models.py`")
    
    with tab4:
        st.header("🔍 Model Explainability")
        
        # Check if explanation results exist
        explain_dir = 'ml_outputs/explanations'
        if os.path.exists(explain_dir):
            st.success("✅ Model explanation results found")
            
            # Load feature importance
            importance_path = os.path.join(explain_dir, 'feature_importance.html')
            if os.path.exists(importance_path):
                st.markdown("### 📊 Feature Importance Analysis")
                
                with open(importance_path, 'r') as f:
                    importance_html = f.read()
                
                st.components.v1.html(importance_html, height=600)
            else:
                st.warning("⚠️ Feature importance analysis not found")
            
            # Load explanation dashboard
            dashboard_path = os.path.join(explain_dir, 'explanation_dashboard.html')
            if os.path.exists(dashboard_path):
                st.markdown("### 🔍 Interactive Explanation Dashboard")
                
                with open(dashboard_path, 'r') as f:
                    dashboard_html = f.read()
                
                st.components.v1.html(dashboard_html, height=800)
            else:
                st.warning("⚠️ Explanation dashboard not found")
        else:
            st.warning("⚠️ No explanation results found. Run the training pipeline first.")
            st.info("To generate explanations, run: `python train_ml_models.py`")
    
    with tab5:
        st.header("📈 Historical Analysis")
        
        # Check if historical data exists
        noaa_dir = 'noaa_data'
        if os.path.exists(noaa_dir):
            st.success("✅ Historical NOAA data found")
            
            # Load some historical data
            try:
                import glob
                csv_files = glob.glob(os.path.join(noaa_dir, "*.csv.gz"))
                
                if csv_files:
                    st.markdown("### 📊 Historical Storm Events")
                    
                    # Load a sample of data
                    sample_file = csv_files[0]
                    df_sample = pd.read_csv(sample_file, compression='gzip', nrows=1000)
                    
                    st.markdown(f"**Sample Data**: {len(df_sample)} events from {os.path.basename(sample_file)}")
                    
                    # Show basic statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📅 Event Distribution by Month")
                        if 'BEGIN_DATE_TIME' in df_sample.columns:
                            df_sample['month'] = pd.to_datetime(df_sample['BEGIN_DATE_TIME'], errors='coerce').dt.month
                            month_counts = df_sample['month'].value_counts().sort_index()
                            
                            fig = px.bar(x=month_counts.index, y=month_counts.values,
                                        title="Storm Events by Month",
                                        labels={'x': 'Month', 'y': 'Event Count'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🌍 Event Distribution by State")
                        if 'STATE' in df_sample.columns:
                            state_counts = df_sample['STATE'].value_counts().head(10)
                            
                            fig = px.bar(x=state_counts.values, y=state_counts.index,
                                        orientation='h',
                                        title="Top 10 States by Event Count",
                                        labels={'x': 'Event Count', 'y': 'State'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data sample
                    st.markdown("#### 📋 Data Sample")
                    st.dataframe(df_sample.head(10), use_container_width=True)
                    
                else:
                    st.warning("⚠️ No CSV files found in NOAA data directory")
                    
            except Exception as e:
                st.error(f"❌ Failed to load historical data: {e}")
        else:
            st.warning("⚠️ Historical NOAA data not found")
            st.info("To download historical data, run the RAG downloader first")

if __name__ == "__main__":
    main()
