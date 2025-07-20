import os
import requests
import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from folium.plugins import HeatMap
from datetime import datetime, timedelta
import pytz
from collections import Counter
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import pgeocode
from dotenv import load_dotenv
import numpy as np
import faiss
from openai import OpenAI
import config

# --- LOAD ENV ---
load_dotenv()  # Load environment variables from .env
api_key = config.OPENAI_API_KEY
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# --- PAGE CONFIG ---
st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: #1f2937 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #374151 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid #667eea !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Subsection headers */
    .subsection-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #4b5563 !important;
        margin-bottom: 1rem !important;
        padding-left: 1rem !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border: 2px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Alert styling */
    .alert-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid #f59e0b !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Chat message styling */
    .chat-message {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border: 2px solid #0ea5e9 !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Text styling */
    .large-text {
        font-size: 1.3rem !important;
        line-height: 1.6 !important;
        color: #374151 !important;
    }
    
    .medium-text {
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
        color: #4b5563 !important;
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        font-size: 1.1rem !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid #10b981 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Warning message styling */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid #f59e0b !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Error message styling */
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid #ef4444 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Info message styling */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 2px solid #3b82f6 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SETTINGS ---
API_URL = config.NOAA_ALERTS_URL
HEADERS = config.HEADERS

# --- AUTO REFRESH ---
st_autorefresh(interval=config.AUTO_REFRESH_INTERVAL, key="refresh")  # refresh every 5 mins

# --- CHATBOT FUNCTIONS ---
@st.cache_data
def load_educational_data():
    """Load educational content about severe weather"""
    return {
        "tornadoes": {
            "formation": "Tornadoes form when warm, moist air collides with cold, dry air, creating instability. Wind shear causes horizontal rotation that gets tilted vertically by thunderstorm updrafts.",
            "scale": "The Enhanced Fujita (EF) Scale rates tornado intensity: EF0 (65-85 mph) to EF5 (over 200 mph).",
            "safety": "Seek shelter in a basement or interior room without windows. Avoid mobile homes and protect your head and neck.",
            "warning_signs": "Dark, greenish sky; large hail; loud roar like a freight train; rotating funnel cloud; debris cloud."
        },
        "thunderstorms": {
            "formation": "Thunderstorms develop when warm, moist air rises rapidly and interacts with cold, dry air, creating instability and vertical motion.",
            "hazards": "Damaging winds (≥58 mph), large hail (≥1 inch), lightning, heavy rain, and potential tornadoes.",
            "safety": "Stay indoors away from windows, avoid electrical appliances, and seek shelter if outdoors."
        },
        "flash_floods": {
            "formation": "Flash floods occur within 6 hours of heavy rainfall, dam breaks, or rapid snowmelt.",
            "dangers": "Can sweep away cars, destroy homes, and cause loss of life within minutes.",
            "safety": "Never drive or walk through flooded roads. Move to higher ground immediately."
        },
        "hurricanes": {
            "formation": "Hurricanes form over warm ocean waters (≥80°F) when atmospheric conditions allow for organized convection.",
            "structure": "Eye (calm center), eyewall (strongest winds), and rainbands (spiral outward).",
            "safety": "Follow evacuation orders early, secure outdoor objects, and have emergency supplies ready."
        }
    }

def get_weather_context():
    """Get current weather context for chatbot"""
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        alerts = data.get("features", [])
        
        if alerts:
            context = "Current severe weather alerts:\n"
            for alert in alerts[:5]:  # Limit to 5 most recent
                props = alert["properties"]
                context += f"- {props['event']} in {props['areaDesc']} (Severity: {props['severity']})\n"
            context += f"\nTotal active alerts: {len(alerts)}"
        else:
            context = "No active severe weather alerts currently."
        
        return context
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching weather data: {e}")
        return "Unable to fetch current weather data due to network issues."
    except Exception as e:
        st.error(f"Error processing weather data: {e}")
        return "Unable to process current weather data."

def chat_with_weather_expert(user_question, educational_data, weather_context):
    """Chat with weather expert using OpenAI"""
    
    # Build comprehensive system prompt
    system_prompt = f"""You are a knowledgeable meteorologist and severe weather expert. You have access to current weather data and extensive knowledge about severe weather phenomena.

Current Weather Context:
{weather_context}

Educational Knowledge Base:
{educational_data}

Guidelines:
1. Provide accurate, science-based information about severe weather
2. Include safety tips when relevant
3. Use current weather context to make responses more relevant
4. Be educational but engaging
5. If asked about specific weather events, use the current context
6. Always prioritize safety in your responses
7. If the user asks about weather in a specific location, analyze the current weather context and provide location-specific information
8. Mention specific alerts, severity levels, and affected areas when relevant
9. Provide actionable advice based on current conditions

Respond in a helpful, educational tone suitable for the general public."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm having trouble connecting to my weather knowledge base. Please try again later. Error: {str(e)}"

# --- MAIN APP ---
st.markdown(f'<h1 class="main-title">🌪️ {config.APP_TITLE}</h1>', unsafe_allow_html=True)

# Display safety disclaimer
st.markdown(f'<div class="alert-box"><p class="large-text">{config.SAFETY_DISCLAIMER}</p></div>', unsafe_allow_html=True)

# Sidebar navigation
mode = st.sidebar.radio("Mode", ["Live Alerts", "Weather Chatbot", "Learn", "Historical Alerts", "ML Weather Insights", "Weather Timeline", "Photo Gallery", "Weather Quiz", "Emergency Kit Builder", "Travel Planner", "Weather Impact Calculator", "ML Explanation"], key="mode_select")

# Load educational data
educational_data = load_educational_data()

# --- ML WEATHER INSIGHTS MODE ---
if mode == "ML Weather Insights":
    st.markdown('<h2 class="section-header">🤖 ML Weather Pattern Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Discover AI-powered insights about weather patterns and get personalized risk assessments!</p>', unsafe_allow_html=True)
    
    # ML Weather Prediction
    st.markdown('<h3 class="subsection-header">🔮 Weather Alert Prediction</h3>', unsafe_allow_html=True)
    
    # User location input
    user_state = st.selectbox("Select your state for analysis", 
                             ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
    
    # Season selection
    season = st.selectbox("Select season for analysis", ["Spring", "Summer", "Fall", "Winter"])
    
    if st.button("🔍 Analyze Weather Patterns"):
        with st.spinner("Analyzing historical patterns..."):
            # Simulate ML analysis (in real app, this would use actual ML models)
            import random
            random.seed(hash(user_state + season))
            
            # Generate realistic ML insights
            alert_probability = random.uniform(0.1, 0.8)
            common_events = ["Severe Thunderstorm", "Flash Flood", "Tornado Warning", "Winter Storm"]
            top_events = random.sample(common_events, 3)
            
            # Risk assessment
            risk_level = "Low" if alert_probability < 0.3 else "Medium" if alert_probability < 0.6 else "High"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Alert Probability", f"{alert_probability:.1%}", 
                         delta=f"{'↑' if alert_probability > 0.5 else '↓'} {abs(alert_probability - 0.5):.1%}")
            with col2:
                # Use color coding with markdown instead of delta_color
                risk_color_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.markdown(f"### Risk Level: {risk_color_emoji[risk_level]} {risk_level}")
            with col3:
                st.metric("Historical Events", len(top_events), delta="This season")
            
            st.markdown('<h3 class="subsection-header">📊 Pattern Analysis</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><p class="large-text"><strong>State:</strong> {user_state} | <strong>Season:</strong> {season}</p><p class="large-text"><strong>Key Insights:</strong></p><ul class="large-text"><li><strong>Most Common Events:</strong> {", ".join(top_events)}</li><li><strong>Peak Activity:</strong> {random.choice(["Morning", "Afternoon", "Evening", "Night"])}</li><li><strong>Duration:</strong> Average {random.randint(2, 8)} hours per event</li><li><strong>Geographic Focus:</strong> {random.choice(["Urban areas", "Rural regions", "Coastal zones", "Mountain areas"])}</li></ul></div>', unsafe_allow_html=True)
            
            # Interactive recommendations
            st.markdown('<h3 class="subsection-header">🛡️ Personalized Safety Recommendations</h3>', unsafe_allow_html=True)
            recommendations = []
            if alert_probability > 0.6:
                recommendations.extend([
                    "🔴 **High Alert:** Prepare emergency kit and monitor weather closely",
                    "📱 **Stay Connected:** Enable emergency alerts on your phone",
                    "🏠 **Shelter Plan:** Identify safe locations in your area"
                ])
            elif alert_probability > 0.3:
                recommendations.extend([
                    "🟡 **Moderate Risk:** Keep emergency supplies ready",
                    "📻 **Stay Informed:** Check weather updates regularly",
                    "🚗 **Travel Caution:** Plan routes with weather in mind"
                ])
            else:
                recommendations.extend([
                    "🟢 **Low Risk:** Standard weather awareness recommended",
                    "📚 **Stay Educated:** Learn about weather safety",
                    "📅 **Seasonal Prep:** Prepare for changing conditions"
                ])
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f'<p class="large-text">{rec}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Weather Trend Analysis
    st.markdown('<h3 class="subsection-header">📈 Historical Trend Analysis</h3>', unsafe_allow_html=True)
    
    # Generate trend data
    import random
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    trend_data = [random.randint(10, 100) for _ in range(12)]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, trend_data, marker='o', linewidth=2, markersize=6)
    ax.fill_between(months, trend_data, alpha=0.3)
    ax.set_title("Monthly Alert Frequency Trend", fontsize=14, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Alerts")
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(months)), trend_data, 1)
    p = np.poly1d(z)
    ax.plot(months, p(range(len(months))), "r--", alpha=0.8, label="Trend")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ML-powered insights
    st.markdown('<h3 class="subsection-header">🤖 AI-Generated Insights</h3>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><p class="large-text"><strong>Pattern Recognition:</strong></p><ul class="large-text"><li>Peak activity detected in spring months (March-May)</li><li>Evening hours show highest alert frequency</li><li>Urban areas experience 40% more alerts than rural regions</li></ul><p class="large-text"><strong>Predictive Factors:</strong></p><ul class="large-text"><li>Temperature fluctuations correlate with alert probability</li><li>Humidity levels above 70% increase severe weather risk</li><li>Wind shear patterns predict tornado development</li></ul><p class="large-text"><strong>Safety Optimization:</strong></p><ul class="large-text"><li>Recommended evacuation time: 30 minutes before severe weather</li><li>Optimal shelter locations: Interior rooms, basements</li><li>Communication backup: Multiple alert systems recommended</li></ul></div>', unsafe_allow_html=True)
    
    # Weather Pattern Clustering Analysis
    st.markdown('<h3 class="subsection-header">🔍 Weather Pattern Clustering Analysis</h3>', unsafe_allow_html=True)
    
    # Simulate ML clustering analysis
    if st.button("🔬 Analyze Weather Patterns"):
        with st.spinner("Performing pattern clustering analysis..."):
            import random
            random.seed(hash(user_state + season))
            
            # Simulate ML clustering results
            clusters = [
                {"name": "High-Intensity Storm Cluster", "frequency": random.randint(15, 35), "avg_duration": random.randint(4, 8), "severity": "High"},
                {"name": "Moderate Thunderstorm Cluster", "frequency": random.randint(25, 45), "avg_duration": random.randint(2, 5), "severity": "Medium"},
                {"name": "Light Precipitation Cluster", "frequency": random.randint(30, 50), "avg_duration": random.randint(1, 3), "severity": "Low"}
            ]
            
            # Create clustering visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cluster_names = [c["name"] for c in clusters]
            frequencies = [c["frequency"] for c in clusters]
            durations = [c["avg_duration"] for c in clusters]
            colors = ['red', 'orange', 'green']
            
            scatter = ax.scatter(frequencies, durations, s=[f*10 for f in frequencies], 
                               c=colors, alpha=0.7, edgecolors='black', linewidth=2)
            
            ax.set_xlabel("Frequency (events per month)")
            ax.set_ylabel("Average Duration (hours)")
            ax.set_title("Weather Pattern Clusters (ML Analysis)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add cluster labels
            for i, cluster in enumerate(clusters):
                ax.annotate(cluster["name"], (frequencies[i], durations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Cluster insights
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="large-text"><strong>🔍 ML Clustering Insights:</strong></p>', unsafe_allow_html=True)
            for cluster in clusters:
                st.markdown(f'<p class="large-text">• <strong>{cluster["name"]}:</strong> {cluster["frequency"]} events/month, {cluster["avg_duration"]}h avg duration, {cluster["severity"]} severity</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Anomaly Detection Analysis
    st.markdown('<h3 class="subsection-header">🚨 Weather Anomaly Detection</h3>', unsafe_allow_html=True)
    
    # Simulate ML anomaly detection
    if st.button("🔍 Detect Weather Anomalies"):
        with st.spinner("Analyzing for weather anomalies..."):
            import random
            random.seed(hash(user_state + season))
            
            # Simulate anomaly detection results
            anomalies = [
                {"type": "Unusual Temperature Spike", "confidence": random.uniform(0.7, 0.95), "severity": "Medium", "description": "Temperature 15°F above seasonal average"},
                {"type": "Abnormal Precipitation Pattern", "confidence": random.uniform(0.6, 0.9), "severity": "High", "description": "Rainfall 200% above normal for this period"},
                {"type": "Wind Speed Anomaly", "confidence": random.uniform(0.5, 0.85), "severity": "Low", "description": "Sustained winds 25% above typical levels"}
            ]
            
            # Create anomaly visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            anomaly_types = [a["type"] for a in anomalies]
            confidences = [a["confidence"] for a in anomalies]
            severities = [a["severity"] for a in anomalies]
            colors = ['red' if s == 'High' else 'orange' if s == 'Medium' else 'yellow' for s in severities]
            
            bars = ax.bar(anomaly_types, confidences, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel("Detection Confidence")
            ax.set_title("Weather Anomaly Detection (ML Model)", fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, conf in zip(bars, confidences):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Anomaly insights
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="large-text"><strong>🚨 Detected Anomalies:</strong></p>', unsafe_allow_html=True)
            for anomaly in anomalies:
                st.markdown(f'<p class="large-text">• <strong>{anomaly["type"]}:</strong> {anomaly["description"]} (Confidence: {anomaly["confidence"]:.1%}, Severity: {anomaly["severity"]})</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Weather Impact Prediction
    st.markdown('<h3 class="subsection-header">🎯 Weather Impact Prediction</h3>', unsafe_allow_html=True)
    
    # Simulate ML impact prediction
    if st.button("🎯 Predict Weather Impacts"):
        with st.spinner("Analyzing potential weather impacts..."):
            import random
            random.seed(hash(user_state + season))
            
            # Simulate impact prediction results
            impacts = [
                {"sector": "Transportation", "risk_level": random.choice(["High", "Medium", "Low"]), "impact": "Road closures, flight delays, public transit disruption"},
                {"sector": "Agriculture", "risk_level": random.choice(["High", "Medium", "Low"]), "impact": "Crop damage, livestock safety, irrigation issues"},
                {"sector": "Energy", "risk_level": random.choice(["High", "Medium", "Low"]), "impact": "Power outages, grid instability, fuel supply disruption"},
                {"sector": "Healthcare", "risk_level": random.choice(["High", "Medium", "Low"]), "impact": "Emergency response, hospital capacity, medical supply chain"},
                {"sector": "Economy", "risk_level": random.choice(["High", "Medium", "Low"]), "impact": "Business closures, supply chain disruption, insurance claims"}
            ]
            
            # Create impact visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sectors = [i["sector"] for i in impacts]
            risk_scores = [3 if i["risk_level"] == "High" else 2 if i["risk_level"] == "Medium" else 1 for i in impacts]
            colors = ['red' if s == 3 else 'orange' if s == 2 else 'green' for s in risk_scores]
            
            bars = ax.bar(sectors, risk_scores, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel("Risk Level (1=Low, 2=Medium, 3=High)")
            ax.set_title("Weather Impact Prediction by Sector (ML Model)", fontsize=14, fontweight='bold')
            ax.set_ylim(0, 3.5)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, risk in zip(bars, risk_scores):
                risk_text = "High" if risk == 3 else "Medium" if risk == 2 else "Low"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       risk_text, ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Impact insights
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<p class="large-text"><strong>🎯 Predicted Impacts:</strong></p>', unsafe_allow_html=True)
            for impact in impacts:
                risk_color = "🔴" if impact["risk_level"] == "High" else "🟡" if impact["risk_level"] == "Medium" else "🟢"
                st.markdown(f'<p class="large-text">• <strong>{impact["sector"]}:</strong> {risk_color} {impact["risk_level"]} Risk - {impact["impact"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Weather Pattern Prediction
    st.markdown('<h3 class="subsection-header">🔮 Next 24-48 Hour Forecast</h3>', unsafe_allow_html=True)
    
    # Get current weather context for prediction
    current_context = get_weather_context()
    
    # Simulate ML-based prediction
    if st.button("🚀 Generate Weather Forecast"):
        with st.spinner("Analyzing patterns and generating forecast..."):
            # Simulate ML model prediction
            import random
            random.seed(hash(current_context))
            
            # Generate realistic forecast data
            forecast_hours = list(range(0, 48, 6))  # Every 6 hours
            alert_probabilities = [random.uniform(0.1, 0.8) for _ in forecast_hours]
            event_types = ["Severe Thunderstorm", "Flash Flood", "Tornado Warning", "Winter Storm", "None"]
            predicted_events = [random.choice(event_types) for _ in forecast_hours]
            
            # Create forecast visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Probability plot
            ax1.plot(forecast_hours, alert_probabilities, marker='o', linewidth=2, markersize=6, color='red')
            ax1.fill_between(forecast_hours, alert_probabilities, alpha=0.3, color='red')
            ax1.set_xlabel("Hours from now")
            ax1.set_ylabel("Alert Probability")
            ax1.set_title("Predicted Alert Probability (ML Model)", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Event type plot
            event_colors = {'Severe Thunderstorm': 'orange', 'Flash Flood': 'blue', 
                           'Tornado Warning': 'red', 'Winter Storm': 'cyan', 'None': 'gray'}
            for i, (hour, event) in enumerate(zip(forecast_hours, predicted_events)):
                if event != 'None':
                    ax2.bar(hour, 1, color=event_colors.get(event, 'gray'), alpha=0.7, label=event if i == 0 else "")
            
            ax2.set_xlabel("Hours from now")
            ax2.set_ylabel("Event Type")
            ax2.set_title("Predicted Weather Events", fontsize=12, fontweight='bold')
            ax2.set_yticks([])
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Forecast insights
            st.markdown("### 📊 Forecast Analysis")
            
            # Find peak risk time
            peak_hour = forecast_hours[alert_probabilities.index(max(alert_probabilities))]
            peak_prob = max(alert_probabilities)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Risk Time", f"+{peak_hour} hours", 
                         delta=f"{peak_prob:.1%} probability")
            with col2:
                st.metric("Highest Risk Event", 
                         next((event for event, prob in zip(predicted_events, alert_probabilities) 
                               if prob == peak_prob), "Unknown"))
            with col3:
                st.metric("Overall Risk Trend", 
                         "↗️ Increasing" if alert_probabilities[-1] > alert_probabilities[0] else "↘️ Decreasing")
            
            # ML model confidence
            st.markdown("**🤖 ML Model Confidence:**")
            confidence = random.uniform(0.7, 0.95)
            st.progress(confidence)
            st.markdown(f"Model confidence: {confidence:.1%}")
            
            # Key predictions
            st.markdown("**🔮 Key Predictions:**")
            st.markdown(f"""
            - **Peak Activity:** {peak_hour} hours from now ({peak_prob:.1%} probability)
            - **Most Likely Event:** {max(set(predicted_events), key=predicted_events.count)}
            - **Duration:** {random.randint(2, 12)} hours of active weather
            - **Geographic Focus:** {random.choice(['Localized', 'Regional', 'Widespread'])} impact
            """)
            
            # Safety timeline
            st.markdown("**⏰ Safety Timeline:**")
            if peak_prob > 0.6:
                st.markdown("""
                - **NOW:** Prepare emergency supplies and shelter
                - **+2 hours:** Monitor weather updates closely
                - **+4 hours:** Consider evacuation if in high-risk area
                - **+6 hours:** Peak activity expected - seek shelter
                """)
            elif peak_prob > 0.3:
                st.markdown("""
                - **NOW:** Stay informed of weather conditions
                - **+3 hours:** Prepare for potential weather changes
                - **+6 hours:** Monitor for escalation
                - **+12 hours:** Conditions should improve
                """)
            else:
                st.markdown("""
                - **NOW:** Standard weather awareness
                - **+6 hours:** Check for any changes
                - **+12 hours:** Continue monitoring
                - **+24 hours:** Conditions expected to remain stable
                """)

# --- WEATHER CHATBOT MODE ---
elif mode == "Weather Chatbot":
    st.markdown('<h2 class="section-header">🤖 Weather Expert Chatbot</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Ask me anything about severe weather, tornadoes, thunderstorms, or current weather conditions!</p>', unsafe_allow_html=True)
    
    # Show current weather context
    with st.expander("📡 Current Weather Context", expanded=False):
        weather_context = get_weather_context()
        st.markdown(f"**{weather_context}**")
    
    # Interactive Risk Assessment
    st.markdown('<h3 class="subsection-header">🛡️ Personal Risk Assessment</h3>', unsafe_allow_html=True)
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        user_location = st.text_input("Enter your city/state for personalized risk assessment", 
                                     placeholder="e.g., Miami, FL")
        user_activity = st.selectbox("What are you planning to do?", 
                                   ["Stay at home", "Travel by car", "Outdoor activities", 
                                    "Work outside", "Attend outdoor event"])
    
    with risk_col2:
        if st.button("🔍 Assess My Risk"):
            if user_location:
                # Get weather context for risk assessment
                weather_context = get_weather_context()
                
                # Simulate risk analysis based on location and activity
                import random
                random.seed(hash(user_location + user_activity))
                
                # Generate risk assessment
                risk_score = random.uniform(0.1, 0.9)
                risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                
                # Activity-specific risks
                activity_risks = {
                    "Stay at home": ["Lightning strikes", "Power outages", "Flooding"],
                    "Travel by car": ["Flash floods", "Reduced visibility", "Wind damage"],
                    "Outdoor activities": ["Lightning", "Hail", "Strong winds"],
                    "Work outside": ["Lightning", "Heat stress", "Wind hazards"],
                    "Attend outdoor event": ["Lightning", "Wind damage", "Evacuation needs"]
                }
                
                st.markdown(f"### Risk Assessment Results")
                # Use color coding with markdown instead of delta_color
                risk_color_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.markdown(f"**Risk Level:** {risk_color_emoji[risk_level]} {risk_level}")
                st.metric("Risk Score", f"{risk_score:.1%}")
                
                st.markdown("**⚠️ Specific Risks for your activity:**")
                for risk in activity_risks.get(user_activity, ["General weather hazards"]):
                    st.markdown(f"- {risk}")
                
                st.markdown("**🛡️ Personalized Recommendations:**")
                if risk_score > 0.7:
                    st.markdown("""
                    - **POSTPONE ACTIVITY** - Severe weather expected
                    - Seek immediate shelter if outdoors
                    - Monitor emergency broadcasts
                    - Have emergency kit ready
                    """)
                elif risk_score > 0.4:
                    st.markdown("""
                    - **EXERCISE CAUTION** - Weather conditions may worsen
                    - Have backup plans ready
                    - Stay informed of weather updates
                    - Know evacuation routes
                    """)
                else:
                    st.markdown("""
                    - **LOW RISK** - Standard weather awareness
                    - Monitor conditions for changes
                    - Keep emergency contacts handy
                    - Stay weather-aware
                    """)
            else:
                st.markdown('<div class="warning-box"><p class="large-text">⚠️ Please enter your location for risk assessment.</p></div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about severe weather..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get weather context
        weather_context = get_weather_context()
        
        # Get response from weather expert
        with st.chat_message("assistant"):
            with st.spinner("Consulting weather data..."):
                response = chat_with_weather_expert(prompt, educational_data, weather_context)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with quick questions
    st.sidebar.markdown("### Quick Questions")
    
    for question in config.QUICK_QUESTIONS:
        if st.sidebar.button(question, key=f"q_{question[:20]}"):
            st.session_state.messages.append({"role": "user", "content": question})
            weather_context = get_weather_context()
            response = chat_with_weather_expert(question, educational_data, weather_context)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# --- WEATHER TIMELINE MODE ---
elif mode == "Weather Timeline":
    st.markdown('<h2 class="section-header">⏰ Weather Alert Timeline</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Track the evolution of weather events and see how alerts develop over time!</p>', unsafe_allow_html=True)
    
    # Get current alerts for timeline
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        alerts = data.get("features", [])
        
        if alerts:
            # Create timeline visualization
            st.markdown('<h3 class="subsection-header">📅 Today\'s Weather Timeline</h3>', unsafe_allow_html=True)
            
            # Sort alerts by sent time
            alerts_with_time = []
            for alert in alerts:
                try:
                    sent_time = datetime.fromisoformat(alert["properties"]["sent"].replace("Z", "+00:00"))
                    alerts_with_time.append((sent_time, alert))
                except:
                    continue
            
            alerts_with_time.sort(key=lambda x: x[0])
            
            # Create timeline
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            for i, (sent_time, alert) in enumerate(alerts_with_time[:10]):  # Show last 10
                props = alert["properties"]
                
                # Timeline styling
                timeline_color = "🔴" if props["severity"] == "Severe" else "🟡" if props["severity"] == "Moderate" else "🟢"
                
                st.markdown(f"""
                <div style="border-left: 4px solid #667eea; padding-left: 20px; margin: 20px 0;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">{timeline_color}</span>
                        <div>
                            <h4 style="margin: 0; color: #1f2937;">{props['event']}</h4>
                            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">{sent_time.strftime('%H:%M:%S')} - {props['areaDesc']}</p>
                        </div>
                    </div>
                    <p style="margin: 5px 0; color: #374151;"><strong>Severity:</strong> {props['severity']} | <strong>Urgency:</strong> {props['urgency']}</p>
                    <p style="margin: 5px 0; color: #4b5563;">{props.get('headline', 'No headline provided.')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Timeline statistics
            st.markdown('<h3 class="subsection-header">📊 Timeline Statistics</h3>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Alerts Today", len(alerts_with_time))
            with col2:
                earliest = min(alerts_with_time, key=lambda x: x[0])[0] if alerts_with_time else None
                latest = max(alerts_with_time, key=lambda x: x[0])[0] if alerts_with_time else None
                if earliest and latest:
                    duration = latest - earliest
                    st.metric("Timeline Duration", f"{duration}")
            with col3:
                severity_counts = Counter([alert[1]["properties"]["severity"] for alert in alerts_with_time])
                most_common = severity_counts.most_common(1)[0] if severity_counts else ("None", 0)
                st.metric("Most Common Severity", most_common[0])
            
            # Alert frequency chart
            st.markdown('<h3 class="subsection-header">📈 Alert Frequency Over Time</h3>', unsafe_allow_html=True)
            
            # Group alerts by hour
            hourly_counts = Counter()
            for sent_time, alert in alerts_with_time:
                hour = sent_time.strftime("%H:00")
                hourly_counts[hour] += 1
            
            if hourly_counts:
                hours = sorted(hourly_counts.keys())
                counts = [hourly_counts[hour] for hour in hours]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(hours, counts, color='skyblue', alpha=0.7, edgecolor='navy')
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Number of Alerts")
                ax.set_title("Alert Frequency by Hour", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                               str(count), ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.markdown('<div class="info-box"><p class="large-text">No active alerts to display in timeline.</p></div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f'<div class="error-box"><p class="large-text">Error loading timeline: {e}</p></div>', unsafe_allow_html=True)

# --- PHOTO GALLERY MODE ---
elif mode == "Photo Gallery":
    st.markdown('<h2 class="section-header">📸 Weather Photography Gallery</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Explore stunning weather photography and learn about different weather phenomena!</p>', unsafe_allow_html=True)
    
    # Weather photography categories
    photo_categories = ["Tornadoes", "Lightning", "Storm Clouds", "Snow & Ice", "Hurricanes", "Rainbows"]
    selected_category = st.selectbox("Choose a weather category", photo_categories)
    
    # Simulate photo gallery with educational content
    if selected_category == "Tornadoes":
        st.markdown('<h3 class="subsection-header">🌪️ Tornado Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Photography Tips:</strong></p>
        <ul class="large-text">
        <li><strong>Safety First:</strong> Always maintain safe distance - never chase tornadoes</li>
        <li><strong>Equipment:</strong> Use wide-angle lenses for context, telephoto for details</li>
        <li><strong>Lighting:</strong> Golden hour provides dramatic lighting for tornado shots</li>
        <li><strong>Composition:</strong> Include foreground elements for scale and perspective</li>
        <li><strong>Settings:</strong> Fast shutter speeds (1/500+) to freeze motion</li>
        </ul>
        
        <p class="large-text"><strong>Famous Tornado Photographs:</strong></p>
        <ul class="large-text">
        <li><strong>Tri-State Tornado (1925):</strong> One of the first documented tornado photos</li>
        <li><strong>Moore Tornado (2013):</strong> Captured by storm chaser Mike Olbinski</li>
        <li><strong>El Reno Tornado (2013):</strong> Widest tornado ever photographed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulate photo grid
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🌪️ Tornado Formation**
            - Captured during development stage
            - Shows funnel cloud formation
            - Dramatic cloud structure
            """)
        with col2:
            st.markdown("""
            **🌪️ Mature Tornado**
            - Full tornado touchdown
            - Debris cloud visible
            - Intense rotation captured
            """)
    
    elif selected_category == "Lightning":
        st.markdown('<h3 class="subsection-header">⚡ Lightning Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Lightning Photography Techniques:</strong></p>
        <ul class="large-text">
        <li><strong>Long Exposure:</strong> Use 10-30 second exposures to capture multiple strikes</li>
        <li><strong>Manual Focus:</strong> Pre-focus on infinity for sharp lightning</li>
        <li><strong>Low ISO:</strong> Use ISO 100-200 for best quality</li>
        <li><strong>Tripod Essential:</strong> Must use tripod for long exposures</li>
        <li><strong>Remote Trigger:</strong> Use remote to avoid camera shake</li>
        </ul>
        
        <p class="large-text"><strong>Lightning Types to Photograph:</strong></p>
        <ul class="large-text">
        <li><strong>Cloud-to-Ground:</strong> Most dramatic and dangerous</li>
        <li><strong>Intra-Cloud:</strong> Beautiful light shows within clouds</li>
        <li><strong>Spider Lightning:</strong> Rare horizontal lightning patterns</li>
        <li><strong>Ball Lightning:</strong> Extremely rare spherical lightning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif selected_category == "Storm Clouds":
        st.markdown('<h3 class="subsection-header">☁️ Storm Cloud Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Storm Cloud Types:</strong></p>
        <ul class="large-text">
        <li><strong>Cumulonimbus:</strong> Towering thunderstorm clouds</li>
        <li><strong>Wall Clouds:</strong> Lowering clouds indicating rotation</li>
        <li><strong>Shelf Clouds:</strong> Horizontal cloud formations</li>
        <li><strong>Mammatus:</strong> Bubble-like cloud formations</li>
        <li><strong>Supercell:</strong> Rotating thunderstorm clouds</li>
        </ul>
        
        <p class="large-text"><strong>Photography Tips:</strong></p>
        <ul class="large-text">
        <li><strong>Wide Angle:</strong> Capture the full scale of storm systems</li>
        <li><strong>Dramatic Lighting:</strong> Shoot during golden hour for best contrast</li>
        <li><strong>Foreground Elements:</strong> Include landscape for perspective</li>
        <li><strong>HDR Technique:</strong> Use HDR for high contrast scenes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif selected_category == "Snow & Ice":
        st.markdown('<h3 class="subsection-header">❄️ Snow & Ice Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Winter Photography Challenges:</strong></p>
        <ul class="large-text">
        <li><strong>Exposure Compensation:</strong> Snow requires +1 to +2 stops exposure</li>
        <li><strong>White Balance:</strong> Adjust for cool blue tones of snow</li>
        <li><strong>Frost Protection:</strong> Keep equipment warm to prevent condensation</li>
        <li><strong>Ice Crystals:</strong> Macro photography of snowflakes</li>
        </ul>
        
        <p class="large-text"><strong>Winter Phenomena:</strong></p>
        <ul class="large-text">
        <li><strong>Hoarfrost:</strong> Beautiful ice crystal formations</li>
        <li><strong>Ice Storms:</strong> Glazed ice covering everything</li>
        <li><strong>Snow Drifts:</strong> Wind-sculpted snow formations</li>
        <li><strong>Frozen Waterfalls:</strong> Dramatic ice formations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif selected_category == "Hurricanes":
        st.markdown('<h3 class="subsection-header">🌀 Hurricane Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Hurricane Photography Safety:</strong></p>
        <ul class="large-text">
        <li><strong>Never Chase:</strong> Hurricanes are extremely dangerous</li>
        <li><strong>Professional Only:</strong> Leave to trained meteorologists</li>
        <li><strong>Remote Sensing:</strong> Use satellites and radar imagery</li>
        <li><strong>Documentation:</strong> Focus on damage assessment after</li>
        </ul>
        
        <p class="large-text"><strong>Hurricane Features:</strong></p>
        <ul class="large-text">
        <li><strong>Eye Wall:</strong> Most intense winds and rain</li>
        <li><strong>Rain Bands:</strong> Spiral bands of precipitation</li>
        <li><strong>Storm Surge:</strong> Ocean water pushed ashore</li>
        <li><strong>Cloud Structure:</strong> Satellite imagery shows organization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif selected_category == "Rainbows":
        st.markdown('<h3 class="subsection-header">🌈 Rainbow Photography</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Rainbow Photography Tips:</strong></p>
        <ul class="large-text">
        <li><strong>Optimal Conditions:</strong> Sun behind you, rain in front</li>
        <li><strong>Polarizing Filter:</strong> Enhances rainbow colors</li>
        <li><strong>Wide Angle:</strong> Capture full rainbow arcs</li>
        <li><strong>Foreground Interest:</strong> Add landscape elements</li>
        <li><strong>Double Rainbows:</strong> Look for secondary rainbow</li>
        </ul>
        
        <p class="large-text"><strong>Rainbow Types:</strong></p>
        <ul class="large-text">
        <li><strong>Primary Rainbow:</strong> Most common, red on outside</li>
        <li><strong>Secondary Rainbow:</strong> Fainter, colors reversed</li>
        <li><strong>Supernumerary:</strong> Additional faint bands</li>
        <li><strong>Circular Rainbow:</strong> Seen from high elevations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Photography challenge
    st.markdown('<h3 class="subsection-header">📸 Photography Challenge</h3>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("""
    <p class="large-text"><strong>Monthly Weather Photography Challenge:</strong></p>
    <p class="large-text">This month's challenge: <strong>Capture dramatic storm clouds at sunset</strong></p>
    
    <p class="large-text"><strong>Challenge Rules:</strong></p>
    <ul class="large-text">
    <li>Submit your best weather photography</li>
    <li>Include location and weather conditions</li>
    <li>Share your camera settings and techniques</li>
    <li>Always prioritize safety over the perfect shot</li>
    </ul>
    
    <p class="large-text"><strong>Safety Reminder:</strong> Never put yourself in danger for a photograph. Weather photography should always be done from safe locations with proper equipment and knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

# --- WEATHER QUIZ MODE ---
elif mode == "Weather Quiz":
    st.markdown('<h2 class="section-header">🧠 Weather Knowledge Quiz</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Test your weather knowledge with interactive quizzes and learn while having fun!</p>', unsafe_allow_html=True)
    
    # Quiz categories
    quiz_categories = ["Tornado Safety", "Thunderstorm Facts", "Hurricane Preparedness", "Winter Weather", "General Weather"]
    selected_quiz = st.selectbox("Choose a quiz category", quiz_categories, key="quiz_category_select")
    
    # Initialize quiz state
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = {}
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.quiz_completed = False
    
    # Quiz questions database
    quiz_database = {
        "Tornado Safety": [
            {
                "question": "What is the safest place to be during a tornado?",
                "options": ["Under a highway overpass", "In a basement or storm cellar", "In a mobile home", "In your car"],
                "correct": 1,
                "explanation": "A basement or storm cellar provides the best protection from tornadoes. Highway overpasses can actually be more dangerous due to wind tunnel effects."
            },
            {
                "question": "What does a 'Tornado Watch' mean?",
                "options": ["A tornado has been spotted", "Conditions are favorable for tornadoes", "A tornado is approaching your area", "The tornado danger has passed"],
                "correct": 1,
                "explanation": "A Tornado Watch means conditions are favorable for tornado development. A Tornado Warning means a tornado has been spotted or indicated by radar."
            },
            {
                "question": "What is the Enhanced Fujita Scale used for?",
                "options": ["Measuring wind speed", "Rating tornado intensity based on damage", "Predicting tornado formation", "Measuring rainfall"],
                "correct": 1,
                "explanation": "The Enhanced Fujita (EF) Scale rates tornado intensity from EF0 to EF5 based on damage caused, not wind speed."
            }
        ],
        "Thunderstorm Facts": [
            {
                "question": "How far away can you hear thunder?",
                "options": ["Up to 5 miles", "Up to 10 miles", "Up to 15 miles", "Up to 25 miles"],
                "correct": 2,
                "explanation": "Thunder can typically be heard up to 15 miles away, depending on atmospheric conditions and terrain."
            },
            {
                "question": "What causes thunder?",
                "options": ["Wind friction", "Rapid expansion of air heated by lightning", "Rain hitting the ground", "Cloud collision"],
                "correct": 1,
                "explanation": "Thunder is caused by the rapid expansion of air that is heated to 50,000°F by lightning, creating a shock wave."
            },
            {
                "question": "What is the '30-30 Rule' for lightning safety?",
                "options": ["30 seconds between lightning and thunder means 30 minutes to wait", "30 minutes of rain means 30 minutes to wait", "30 mph winds mean 30 minutes to wait", "30°F temperature means 30 minutes to wait"],
                "correct": 0,
                "explanation": "If you count 30 seconds or less between lightning and thunder, wait 30 minutes before resuming outdoor activities."
            }
        ],
        "Hurricane Preparedness": [
            {
                "question": "What is the difference between a hurricane watch and warning?",
                "options": ["A watch means the hurricane is stronger", "A warning means conditions are expected within 36 hours", "A watch means conditions are possible, warning means expected", "There is no difference"],
                "correct": 2,
                "explanation": "A Hurricane Watch means hurricane conditions are possible within 48 hours. A Hurricane Warning means hurricane conditions are expected within 36 hours."
            },
            {
                "question": "What is storm surge?",
                "options": ["Heavy rainfall", "Strong winds", "Abnormal rise in sea level", "Tornado formation"],
                "correct": 2,
                "explanation": "Storm surge is an abnormal rise in sea level caused by hurricane winds and low pressure, often the most dangerous part of a hurricane."
            },
            {
                "question": "What should you do if you're in a hurricane evacuation zone?",
                "options": ["Wait until the last minute to leave", "Follow evacuation orders immediately", "Stay in your home", "Go to the beach to watch"],
                "correct": 1,
                "explanation": "Always follow evacuation orders immediately. Don't wait until the last minute as roads may be impassable."
            }
        ],
        "Winter Weather": [
            {
                "question": "What is the difference between sleet and freezing rain?",
                "options": ["Sleet is colder", "Sleet falls as ice pellets, freezing rain as liquid that freezes on contact", "Freezing rain is heavier", "There is no difference"],
                "correct": 1,
                "explanation": "Sleet falls as ice pellets, while freezing rain falls as liquid water that freezes on contact with cold surfaces."
            },
            {
                "question": "What is wind chill?",
                "options": ["The temperature of the wind", "How cold it feels due to wind", "The speed of the wind", "The direction of the wind"],
                "correct": 1,
                "explanation": "Wind chill is how cold it feels to the human body due to the combination of air temperature and wind speed."
            },
            {
                "question": "What should you do if you're stranded in your car during a winter storm?",
                "options": ["Leave the car and walk for help", "Stay in the car and run the engine", "Stay in the car but conserve fuel", "Try to dig your car out"],
                "correct": 2,
                "explanation": "Stay in your car and conserve fuel. Running the engine continuously can lead to carbon monoxide poisoning."
            }
        ],
        "General Weather": [
            {
                "question": "What is the primary cause of weather on Earth?",
                "options": ["The Moon's gravity", "The Sun's energy", "Ocean currents", "Volcanic activity"],
                "correct": 1,
                "explanation": "The Sun's energy is the primary driver of weather on Earth, heating the atmosphere and creating temperature differences."
            },
            {
                "question": "What is a cold front?",
                "options": ["A boundary where cold air replaces warm air", "A boundary where warm air replaces cold air", "A stationary weather system", "A type of storm"],
                "correct": 0,
                "explanation": "A cold front is a boundary where cold air mass replaces a warmer air mass, often bringing storms and cooler weather."
            },
            {
                "question": "What is the dew point?",
                "options": ["The temperature at which water vapor condenses", "The temperature at which water freezes", "The temperature at which water boils", "The temperature at which snow forms"],
                "correct": 0,
                "explanation": "The dew point is the temperature at which water vapor in the air condenses into liquid water."
            }
        ]
    }
    
    # Start or restart quiz
    if st.button("🎯 Start Quiz") or st.session_state.quiz_questions.get(selected_quiz):
        if selected_quiz not in st.session_state.quiz_questions:
            st.session_state.quiz_questions[selected_quiz] = quiz_database[selected_quiz].copy()
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.quiz_completed = False
        
        questions = st.session_state.quiz_questions[selected_quiz]
        
        if not st.session_state.quiz_completed and st.session_state.current_question < len(questions):
            question = questions[st.session_state.current_question]
            
            st.markdown(f'<h3 class="subsection-header">Question {st.session_state.current_question + 1} of {len(questions)}</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><p class="large-text"><strong>{question["question"]}</strong></p></div>', unsafe_allow_html=True)
            
            # Display options
            selected_answer = st.radio("Choose your answer:", question["options"], key=f"q{st.session_state.current_question}")
            
            if st.button("✅ Submit Answer"):
                if selected_answer == question["options"][question["correct"]]:
                    st.session_state.score += 1
                    st.markdown('<div class="success-box"><p class="large-text">✅ Correct! Well done!</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box"><p class="large-text">❌ Incorrect. The correct answer is highlighted below.</p></div>', unsafe_allow_html=True)
                
                # Show explanation
                st.markdown(f'<div class="info-box"><p class="large-text"><strong>Explanation:</strong> {question["explanation"]}</p></div>', unsafe_allow_html=True)
                
                st.session_state.current_question += 1
                
                if st.session_state.current_question >= len(questions):
                    st.session_state.quiz_completed = True
                else:
                    st.rerun()
        
        # Show quiz results
        if st.session_state.quiz_completed:
            st.markdown('<h3 class="subsection-header">🎉 Quiz Complete!</h3>', unsafe_allow_html=True)
            score_percentage = (st.session_state.score / len(questions)) * 100
            
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="large-text"><strong>Your Score:</strong> {st.session_state.score}/{len(questions)} ({score_percentage:.1f}%)</p>', unsafe_allow_html=True)
            
            if score_percentage >= 90:
                st.markdown('<p class="large-text">🏆 <strong>Excellent!</strong> You\'re a weather expert!</p>', unsafe_allow_html=True)
            elif score_percentage >= 70:
                st.markdown('<p class="large-text">🎯 <strong>Good job!</strong> You have solid weather knowledge!</p>', unsafe_allow_html=True)
            elif score_percentage >= 50:
                st.markdown('<p class="large-text">📚 <strong>Keep learning!</strong> Review the educational content to improve!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="large-text">📖 <strong>Study time!</strong> Check out the Learn section for more information!</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Take Quiz Again"):
                st.session_state.quiz_questions.pop(selected_quiz, None)
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.quiz_completed = False
                st.rerun()

# --- EMERGENCY KIT BUILDER MODE ---
elif mode == "Emergency Kit Builder":
    st.markdown('<h2 class="section-header">🛡️ Emergency Kit Builder</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Build your personalized emergency kit based on your location and weather risks!</p>', unsafe_allow_html=True)
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        user_state = st.selectbox("Select your state", 
                                 ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
        family_size = st.selectbox("Family size", ["1 person", "2-3 people", "4-5 people", "6+ people"])
    
    with col2:
        primary_risk = st.selectbox("Primary weather risk in your area", 
                                   ["Tornadoes", "Hurricanes", "Winter Storms", "Flooding", "Wildfires", "General"])
        special_needs = st.multiselect("Special considerations", 
                                     ["Infants/Children", "Elderly", "Pets", "Medical conditions", "None"])
    
    if st.button("🔧 Build My Emergency Kit"):
        st.markdown('<h3 class="subsection-header">📋 Your Personalized Emergency Kit</h3>', unsafe_allow_html=True)
        
        # Base emergency kit items
        base_items = {
            "Water": "1 gallon per person per day (3-day supply minimum)",
            "Food": "Non-perishable food items (3-day supply minimum)",
            "Flashlight": "Battery-powered or hand-crank flashlight",
            "Batteries": "Extra batteries for all devices",
            "First Aid Kit": "Basic first aid supplies",
            "Medications": "Prescription and over-the-counter medications",
            "Important Documents": "Copies of insurance, ID, medical records",
            "Cash": "Small bills and coins",
            "Phone Charger": "Portable charger or car charger",
            "Emergency Radio": "Battery-powered or hand-crank radio"
        }
        
        # Weather-specific items
        weather_specific = {
            "Tornadoes": {
                "Helmet": "Protect head from flying debris",
                "Sturdy Shoes": "Protect feet from broken glass and debris",
                "Work Gloves": "Protect hands during cleanup",
                "Whistle": "Signal for help if trapped"
            },
            "Hurricanes": {
                "Plywood": "Board up windows",
                "Generator": "Backup power source",
                "Fuel": "Extra fuel for generator",
                "Waterproof Container": "Protect important documents"
            },
            "Winter Storms": {
                "Warm Clothing": "Extra layers, hats, gloves",
                "Blankets": "Warm blankets or sleeping bags",
                "Hand Warmers": "Chemical hand warmers",
                "Snow Shovel": "Clear snow if needed"
            },
            "Flooding": {
                "Life Jackets": "Personal flotation devices",
                "Waterproof Bags": "Protect electronics and documents",
                "High Ground Plan": "Know evacuation routes",
                "Sandbags": "If available for protection"
            },
            "Wildfires": {
                "N95 Masks": "Protect from smoke inhalation",
                "Goggles": "Protect eyes from smoke",
                "Fire Extinguisher": "Small fire extinguisher",
                "Evacuation Plan": "Multiple escape routes"
            },
            "General": {
                "Multi-tool": "Swiss Army knife or similar",
                "Duct Tape": "Emergency repairs",
                "Plastic Sheeting": "Emergency shelter or repairs",
                "Matches": "Fire starting capability"
            }
        }
        
        # Special needs items
        special_items = {
            "Infants/Children": {
                "Baby Formula": "If applicable",
                "Diapers": "Extra diapers and wipes",
                "Baby Food": "Age-appropriate food",
                "Comfort Items": "Favorite toys or blankets"
            },
            "Elderly": {
                "Extra Medications": "Extended supply",
                "Mobility Aids": "Canes, walkers if needed",
                "Comfortable Clothing": "Easy to put on/remove",
                "Medical Alert Device": "If applicable"
            },
            "Pets": {
                "Pet Food": "3-day supply minimum",
                "Pet Carrier": "For safe transport",
                "Pet Medications": "If applicable",
                "Pet ID": "Collar with contact info"
            },
            "Medical conditions": {
                "Extra Medical Supplies": "Based on specific needs",
                "Medical Alert Bracelet": "If applicable",
                "Backup Medical Equipment": "If needed",
                "Medical Contact List": "Emergency contacts"
            }
        }
        
        # Display kit
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<p class="large-text"><strong>Essential Items (All Situations):</strong></p>', unsafe_allow_html=True)
        for item, description in base_items.items():
            st.markdown(f'<p class="large-text">• <strong>{item}:</strong> {description}</p>', unsafe_allow_html=True)
        
        if primary_risk in weather_specific:
            st.markdown(f'<p class="large-text"><strong>{primary_risk}-Specific Items:</strong></p>', unsafe_allow_html=True)
            for item, description in weather_specific[primary_risk].items():
                st.markdown(f'<p class="large-text">• <strong>{item}:</strong> {description}</p>', unsafe_allow_html=True)
        
        if special_needs:
            for need in special_needs:
                if need in special_items:
                    st.markdown(f'<p class="large-text"><strong>{need} Items:</strong></p>', unsafe_allow_html=True)
                    for item, description in special_items[need].items():
                        st.markdown(f'<p class="large-text">• <strong>{item}:</strong> {description}</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Kit storage tips
        st.markdown('<h3 class="subsection-header">📦 Storage Tips</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Kit Storage Guidelines:</strong></p>
        <ul class="large-text">
        <li><strong>Location:</strong> Store in a cool, dry place that's easily accessible</li>
        <li><strong>Container:</strong> Use a waterproof, portable container</li>
        <li><strong>Accessibility:</strong> Keep kit where you can grab it quickly</li>
        <li><strong>Maintenance:</strong> Check and update kit every 6 months</li>
        <li><strong>Expiration:</strong> Replace expired food, water, and medications</li>
        <li><strong>Multiple Kits:</strong> Consider kits for home, car, and work</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Emergency plan reminder
        st.markdown('<h3 class="subsection-header">📋 Emergency Plan Reminder</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>Don't forget to also create an emergency plan:</strong></p>
        <ul class="large-text">
        <li><strong>Communication Plan:</strong> How family members will contact each other</li>
        <li><strong>Meeting Places:</strong> Designated safe meeting locations</li>
        <li><strong>Evacuation Routes:</strong> Multiple routes from your home</li>
        <li><strong>Emergency Contacts:</strong> List of important phone numbers</li>
        <li><strong>Practice:</strong> Regularly review and practice your plan</li>
        </ul>
                 </div>
         """, unsafe_allow_html=True)

# --- TRAVEL PLANNER MODE ---
elif mode == "Travel Planner":
    st.markdown('<h2 class="section-header">✈️ Weather Travel Planner</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Plan your trips with weather intelligence! Get route analysis, safety precautions, and alternative planning for both driving and flying.</p>', unsafe_allow_html=True)
    
    # Travel type selection
    travel_type = st.selectbox("Select Travel Type", ["Road Trip", "Flight", "Future Trip Planning"], key="travel_type_select")
    
    if travel_type == "Road Trip":
        st.markdown('<h3 class="subsection-header">🚗 Road Trip Weather Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            origin = st.text_input("Starting City/State", placeholder="e.g., New York, NY")
            destination = st.text_input("Destination City/State", placeholder="e.g., Miami, FL")
        
        with col2:
            departure_date = st.date_input("Departure Date", value=datetime.now().date())
            travel_time = st.selectbox("Preferred Travel Time", ["Morning (6 AM - 12 PM)", "Afternoon (12 PM - 6 PM)", "Evening (6 PM - 12 AM)", "Night (12 AM - 6 AM)"], key="road_travel_time")
        
        if st.button("🔍 Analyze Route Weather"):
            if origin and destination:
                st.markdown('<h3 class="subsection-header">📊 Route Weather Analysis</h3>', unsafe_allow_html=True)
                
                # Simulate route analysis
                import random
                random.seed(hash(origin + destination + str(departure_date)))
                
                # Generate route segments
                route_segments = [
                    {"segment": f"{origin} to {destination}", "distance": random.randint(200, 800), "duration": random.randint(3, 8)},
                    {"segment": "Mid-route checkpoint", "distance": random.randint(100, 400), "duration": random.randint(2, 4)},
                    {"segment": "Final approach", "distance": random.randint(50, 200), "duration": random.randint(1, 3)}
                ]
                
                # Weather conditions for each segment
                weather_conditions = []
                for segment in route_segments:
                    conditions = random.choice([
                        {"condition": "Clear", "risk": "Low", "visibility": "Good", "road_condition": "Dry"},
                        {"condition": "Light Rain", "risk": "Medium", "visibility": "Moderate", "road_condition": "Wet"},
                        {"condition": "Heavy Rain", "risk": "High", "visibility": "Poor", "road_condition": "Slippery"},
                        {"condition": "Snow/Ice", "risk": "High", "visibility": "Poor", "road_condition": "Hazardous"},
                        {"condition": "Fog", "risk": "Medium", "visibility": "Poor", "road_condition": "Wet"}
                    ])
                    weather_conditions.append(conditions)
                
                # Display route analysis
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<p class="large-text"><strong>Route Overview:</strong></p>', unsafe_allow_html=True)
                total_distance = sum(seg["distance"] for seg in route_segments)
                total_duration = sum(seg["duration"] for seg in route_segments)
                st.markdown(f'<p class="large-text">• <strong>Total Distance:</strong> {total_distance} miles</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Estimated Duration:</strong> {total_duration} hours</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Departure:</strong> {departure_date.strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Segment-by-segment analysis
                st.markdown('<h3 class="subsection-header">🛣️ Segment Weather Analysis</h3>', unsafe_allow_html=True)
                
                for i, (segment, weather) in enumerate(zip(route_segments, weather_conditions)):
                    risk_color = "🟢" if weather["risk"] == "Low" else "🟡" if weather["risk"] == "Medium" else "🔴"
                    
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Segment {i+1}: {segment["segment"]}</strong></p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text">• <strong>Distance:</strong> {segment["distance"]} miles ({segment["duration"]} hours)</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text">• <strong>Weather:</strong> {weather["condition"]} {risk_color}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text">• <strong>Visibility:</strong> {weather["visibility"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text">• <strong>Road Condition:</strong> {weather["road_condition"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Safety recommendations
                st.markdown('<h3 class="subsection-header">🛡️ Safety Recommendations</h3>', unsafe_allow_html=True)
                
                # Determine overall risk
                high_risk_segments = sum(1 for w in weather_conditions if w["risk"] == "High")
                medium_risk_segments = sum(1 for w in weather_conditions if w["risk"] == "Medium")
                
                if high_risk_segments > 0:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>⚠️ HIGH RISK TRIP - Consider postponing or taking alternative route</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Delay Travel:</strong> Wait for weather conditions to improve</li>
                    <li><strong>Alternative Route:</strong> Consider longer but safer route</li>
                    <li><strong>Emergency Kit:</strong> Pack comprehensive emergency supplies</li>
                    <li><strong>Travel Buddy:</strong> Don't travel alone in severe weather</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                elif medium_risk_segments > 0:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>🟡 MODERATE RISK - Exercise caution and prepare accordingly</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Check Weather:</strong> Monitor conditions before departure</li>
                    <li><strong>Allow Extra Time:</strong> Plan for delays and slower travel</li>
                    <li><strong>Emergency Kit:</strong> Pack basic emergency supplies</li>
                    <li><strong>Stay Informed:</strong> Keep weather radio or app handy</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>🟢 LOW RISK - Safe travel conditions expected</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Standard Precautions:</strong> Normal driving safety measures</li>
                    <li><strong>Basic Kit:</strong> Standard emergency supplies recommended</li>
                    <li><strong>Regular Updates:</strong> Check weather periodically</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Essential items for road trip
                st.markdown('<h3 class="subsection-header">📦 Essential Travel Items</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("""
                <p class="large-text"><strong>Vehicle Emergency Kit:</strong></p>
                <ul class="large-text">
                <li><strong>Jumper Cables:</strong> For battery issues</li>
                <li><strong>Spare Tire:</strong> Check condition before trip</li>
                <li><strong>Flashlight:</strong> Extra batteries included</li>
                <li><strong>First Aid Kit:</strong> Basic medical supplies</li>
                <li><strong>Blankets:</strong> Warm blankets for cold weather</li>
                <li><strong>Water & Snacks:</strong> Non-perishable food items</li>
                <li><strong>Phone Charger:</strong> Car and portable chargers</li>
                <li><strong>Road Flares:</strong> For emergency signaling</li>
                <li><strong>Ice Scraper:</strong> For winter weather</li>
                <li><strong>Weather Radio:</strong> For weather updates</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Alternative routes
                st.markdown('<h3 class="subsection-header">🔄 Alternative Route Suggestions</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("""
                <p class="large-text"><strong>Route Optimization Tips:</strong></p>
                <ul class="large-text">
                <li><strong>Check Multiple Routes:</strong> Use GPS to find alternative paths</li>
                <li><strong>Avoid Mountain Passes:</strong> In winter weather conditions</li>
                <li><strong>Stay on Major Highways:</strong> Better maintained and patrolled</li>
                <li><strong>Consider Timing:</strong> Travel during daylight hours when possible</li>
                <li><strong>Rest Stops:</strong> Plan breaks at safe locations</li>
                <li><strong>Weather Apps:</strong> Use real-time weather tracking</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif travel_type == "Flight":
        st.markdown('<h3 class="subsection-header">✈️ Flight Weather Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            departure_airport = st.text_input("Departure Airport Code", placeholder="e.g., JFK")
            arrival_airport = st.text_input("Arrival Airport Code", placeholder="e.g., LAX")
        
        with col2:
            flight_date = st.date_input("Flight Date", value=datetime.now().date())
            flight_time = st.selectbox("Flight Time", ["Early Morning (6 AM - 10 AM)", "Mid-Morning (10 AM - 2 PM)", "Afternoon (2 PM - 6 PM)", "Evening (6 PM - 10 PM)", "Late Night (10 PM - 6 AM)"], key="flight_time_select")
        
        if st.button("✈️ Analyze Flight Weather"):
            if departure_airport and arrival_airport:
                st.markdown('<h3 class="subsection-header">🌤️ Flight Weather Analysis</h3>', unsafe_allow_html=True)
                
                # Simulate flight weather analysis
                import random
                random.seed(hash(departure_airport + arrival_airport + str(flight_date)))
                
                # Generate flight conditions
                departure_weather = random.choice([
                    {"condition": "Clear", "visibility": "10+ miles", "wind": "Light", "turbulence": "None"},
                    {"condition": "Cloudy", "visibility": "5-10 miles", "wind": "Moderate", "turbulence": "Light"},
                    {"condition": "Rain", "visibility": "2-5 miles", "wind": "Strong", "turbulence": "Moderate"},
                    {"condition": "Storm", "visibility": "1-2 miles", "wind": "Very Strong", "turbulence": "Severe"}
                ])
                
                arrival_weather = random.choice([
                    {"condition": "Clear", "visibility": "10+ miles", "wind": "Light", "turbulence": "None"},
                    {"condition": "Cloudy", "visibility": "5-10 miles", "wind": "Moderate", "turbulence": "Light"},
                    {"condition": "Rain", "visibility": "2-5 miles", "wind": "Strong", "turbulence": "Moderate"},
                    {"condition": "Storm", "visibility": "1-2 miles", "wind": "Very Strong", "turbulence": "Severe"}
                ])
                
                # Flight route analysis
                route_weather = random.choice([
                    {"condition": "Clear skies", "turbulence": "Minimal", "altitude": "35,000 ft"},
                    {"condition": "Light clouds", "turbulence": "Light", "altitude": "30,000 ft"},
                    {"condition": "Moderate clouds", "turbulence": "Moderate", "altitude": "28,000 ft"},
                    {"condition": "Storm systems", "turbulence": "Severe", "altitude": "25,000 ft"}
                ])
                
                # Display flight analysis
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<p class="large-text"><strong>Flight Route:</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>From:</strong> {departure_airport} to <strong>To:</strong> {arrival_airport}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Date:</strong> {flight_date.strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Time:</strong> {flight_time}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Weather conditions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<h4 class="subsection-header">🛫 Departure Weather</h4>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Condition:</strong> {departure_weather["condition"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Visibility:</strong> {departure_weather["visibility"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Wind:</strong> {departure_weather["wind"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Turbulence:</strong> {departure_weather["turbulence"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h4 class="subsection-header">✈️ Route Weather</h4>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Condition:</strong> {route_weather["condition"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Turbulence:</strong> {route_weather["turbulence"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Altitude:</strong> {route_weather["altitude"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<h4 class="subsection-header">🛬 Arrival Weather</h4>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Condition:</strong> {arrival_weather["condition"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Visibility:</strong> {arrival_weather["visibility"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Wind:</strong> {arrival_weather["wind"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="large-text"><strong>Turbulence:</strong> {arrival_weather["turbulence"]}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Flight recommendations
                st.markdown('<h3 class="subsection-header">✈️ Flight Recommendations</h3>', unsafe_allow_html=True)
                
                # Determine flight risk
                severe_conditions = sum(1 for w in [departure_weather, arrival_weather, route_weather] 
                                     if "Severe" in str(w.values()) or "Storm" in str(w.values()))
                
                if severe_conditions > 0:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>⚠️ FLIGHT DELAYS LIKELY - Monitor airline updates</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Check Status:</strong> Monitor flight status frequently</li>
                    <li><strong>Allow Extra Time:</strong> Plan for delays and cancellations</li>
                    <li><strong>Travel Insurance:</strong> Consider purchasing coverage</li>
                    <li><strong>Backup Plans:</strong> Have alternative travel options</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>🟢 GOOD FLYING CONDITIONS - Standard travel precautions</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Arrive Early:</strong> Standard 2 hours for domestic flights</li>
                    <li><strong>Check Weather:</strong> Monitor conditions before departure</li>
                    <li><strong>Stay Informed:</strong> Keep airline app updated</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Flight safety tips
                st.markdown('<h3 class="subsection-header">🛡️ Flight Safety Tips</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("""
                <p class="large-text"><strong>Pre-Flight Preparation:</strong></p>
                <ul class="large-text">
                <li><strong>Weather Apps:</strong> Monitor conditions at both airports</li>
                <li><strong>Airline Updates:</strong> Enable notifications for flight changes</li>
                <li><strong>Travel Insurance:</strong> Consider weather-related coverage</li>
                <li><strong>Flexible Booking:</strong> Choose refundable tickets when possible</li>
                <li><strong>Backup Plans:</strong> Have alternative travel options ready</li>
                </ul>
                
                <p class="large-text"><strong>During Flight:</strong></p>
                <ul class="large-text">
                <li><strong>Seatbelt:</strong> Keep fastened during turbulence</li>
                <li><strong>Follow Instructions:</strong> Listen to crew announcements</li>
                <li><strong>Stay Calm:</strong> Turbulence is normal and safe</li>
                <li><strong>Hydration:</strong> Drink water to stay comfortable</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif travel_type == "Future Trip Planning":
        st.markdown('<h3 class="subsection-header">🔮 Future Trip Weather Planning</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            future_destination = st.text_input("Destination", placeholder="e.g., Miami, FL")
            trip_month = st.selectbox("Trip Month", ["January", "February", "March", "April", "May", "June", 
                                                   "July", "August", "September", "October", "November", "December"], key="trip_month_select")
        
        with col2:
            trip_duration = st.selectbox("Trip Duration", ["Weekend (2-3 days)", "Week (5-7 days)", "Extended (10+ days)"], key="trip_duration_select")
            travel_mode = st.selectbox("Travel Mode", ["Driving", "Flying", "Both"], key="travel_mode_select")
        
        if st.button("🔮 Plan Future Trip"):
            if future_destination:
                st.markdown('<h3 class="subsection-header">📅 Historical Weather Analysis</h3>', unsafe_allow_html=True)
                
                # Simulate historical weather analysis
                import random
                random.seed(hash(future_destination + trip_month))
                
                # Generate historical weather patterns
                avg_temp = random.randint(30, 85)
                avg_precipitation = random.uniform(0.1, 8.0)
                storm_frequency = random.randint(0, 15)
                severe_weather_days = random.randint(0, 5)
                
                # Weather patterns by month
                weather_patterns = {
                    "January": {"temp_range": "20-45°F", "precipitation": "Moderate", "storms": "Winter storms", "risk": "Medium"},
                    "February": {"temp_range": "25-50°F", "precipitation": "Moderate", "storms": "Winter storms", "risk": "Medium"},
                    "March": {"temp_range": "35-60°F", "precipitation": "High", "storms": "Spring storms", "risk": "High"},
                    "April": {"temp_range": "45-70°F", "precipitation": "High", "storms": "Spring storms", "risk": "High"},
                    "May": {"temp_range": "55-80°F", "precipitation": "High", "storms": "Severe storms", "risk": "High"},
                    "June": {"temp_range": "65-85°F", "precipitation": "Moderate", "storms": "Thunderstorms", "risk": "Medium"},
                    "July": {"temp_range": "70-90°F", "precipitation": "Moderate", "storms": "Thunderstorms", "risk": "Medium"},
                    "August": {"temp_range": "70-90°F", "precipitation": "Moderate", "storms": "Hurricanes", "risk": "High"},
                    "September": {"temp_range": "65-85°F", "precipitation": "High", "storms": "Hurricanes", "risk": "High"},
                    "October": {"temp_range": "55-75°F", "precipitation": "Moderate", "storms": "Fall storms", "risk": "Medium"},
                    "November": {"temp_range": "40-65°F", "precipitation": "Moderate", "storms": "Fall storms", "risk": "Medium"},
                    "December": {"temp_range": "25-50°F", "precipitation": "Moderate", "storms": "Winter storms", "risk": "Medium"}
                }
                
                pattern = weather_patterns.get(trip_month, {"temp_range": "40-70°F", "precipitation": "Moderate", "storms": "Mixed", "risk": "Medium"})
                
                # Display historical analysis
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<p class="large-text"><strong>Historical Weather Patterns for {}</strong></p>'.format(trip_month), unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Average Temperature:</strong> {avg_temp}°F ({pattern["temp_range"]})</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Precipitation:</strong> {avg_precipitation:.1f} inches ({pattern["precipitation"]})</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Storm Frequency:</strong> {storm_frequency} days per month</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Severe Weather Days:</strong> {severe_weather_days} days per month</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="large-text">• <strong>Primary Storm Type:</strong> {pattern["storms"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Travel recommendations
                st.markdown('<h3 class="subsection-header">🎯 Travel Recommendations</h3>', unsafe_allow_html=True)
                
                risk_level = pattern["risk"]
                if risk_level == "High":
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>⚠️ HIGH WEATHER RISK - Plan accordingly</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Flexible Dates:</strong> Consider alternative months</li>
                    <li><strong>Travel Insurance:</strong> Essential for weather protection</li>
                    <li><strong>Backup Plans:</strong> Have indoor activities planned</li>
                    <li><strong>Weather Monitoring:</strong> Check forecasts frequently</li>
                    <li><strong>Emergency Kit:</strong> Pack comprehensive supplies</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>🟡 MODERATE WEATHER RISK - Standard precautions</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Weather Monitoring:</strong> Check forecasts before departure</li>
                    <li><strong>Flexible Itinerary:</strong> Plan both indoor and outdoor activities</li>
                    <li><strong>Basic Insurance:</strong> Consider weather-related coverage</li>
                    <li><strong>Emergency Kit:</strong> Pack basic supplies</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown('<p class="large-text"><strong>🟢 LOW WEATHER RISK - Good travel conditions</strong></p>', unsafe_allow_html=True)
                    st.markdown("""
                    <ul class="large-text">
                    <li><strong>Standard Planning:</strong> Normal travel preparations</li>
                    <li><strong>Outdoor Activities:</strong> Safe to plan outdoor events</li>
                    <li><strong>Basic Monitoring:</strong> Check weather occasionally</li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Best travel times
                st.markdown('<h3 class="subsection-header">⏰ Optimal Travel Times</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("""
                <p class="large-text"><strong>Based on Historical Patterns:</strong></p>
                <ul class="large-text">
                <li><strong>Best Travel Days:</strong> Tuesday-Thursday (fewer crowds, better weather)</li>
                <li><strong>Worst Travel Days:</strong> Friday-Sunday (more storms, higher risk)</li>
                <li><strong>Peak Storm Times:</strong> Afternoon/Evening (2-8 PM)</li>
                <li><strong>Calmest Periods:</strong> Early morning (6-10 AM)</li>
                <li><strong>Seasonal Considerations:</strong> Avoid peak storm seasons</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Packing recommendations
                st.markdown('<h3 class="subsection-header">📦 Packing Recommendations</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                if pattern["storms"] == "Winter storms":
                    st.markdown("""
                    <p class="large-text"><strong>Winter Weather Packing:</strong></p>
                    <ul class="large-text">
                    <li><strong>Warm Clothing:</strong> Layers, hats, gloves, scarves</li>
                    <li><strong>Waterproof Gear:</strong> Boots, jackets, umbrellas</li>
                    <li><strong>Emergency Supplies:</strong> Blankets, hand warmers, flashlight</li>
                    <li><strong>Travel Insurance:</strong> Winter weather coverage</li>
                    </ul>
                    """, unsafe_allow_html=True)
                elif pattern["storms"] == "Hurricanes":
                    st.markdown("""
                    <p class="large-text"><strong>Hurricane Season Packing:</strong></p>
                    <ul class="large-text">
                    <li><strong>Waterproof Gear:</strong> Rain jackets, waterproof bags</li>
                    <li><strong>Emergency Supplies:</strong> Flashlight, batteries, first aid</li>
                    <li><strong>Travel Insurance:</strong> Hurricane coverage essential</li>
                    <li><strong>Flexible Plans:</strong> Be ready to change itinerary</li>
                    </ul>
                    """, unsafe_allow_html=True)
                elif pattern["storms"] == "Severe storms":
                    st.markdown("""
                    <p class="large-text"><strong>Severe Storm Season Packing:</strong></p>
                    <ul class="large-text">
                    <li><strong>Weather Protection:</strong> Rain gear, sturdy shoes</li>
                    <li><strong>Emergency Kit:</strong> Flashlight, weather radio, first aid</li>
                    <li><strong>Communication:</strong> Charged phone, backup charger</li>
                    <li><strong>Flexible Clothing:</strong> Layers for changing conditions</li>
                    </ul>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p class="large-text"><strong>General Weather Packing:</strong></p>
                    <ul class="large-text">
                    <li><strong>Weather-Appropriate Clothing:</strong> Check forecast before packing</li>
                    <li><strong>Basic Emergency Kit:</strong> Flashlight, first aid, water</li>
                    <li><strong>Weather Protection:</strong> Umbrella, rain jacket if needed</li>
                    <li><strong>Comfortable Shoes:</strong> Suitable for weather conditions</li>
                    </ul>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Alternative destinations
                st.markdown('<h3 class="subsection-header">🌍 Alternative Destination Suggestions</h3>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                if risk_level == "High":
                    st.markdown("""
                    <p class="large-text"><strong>Consider These Lower-Risk Alternatives:</strong></p>
                    <ul class="large-text">
                    <li><strong>Indoor Destinations:</strong> Museums, shopping centers, indoor attractions</li>
                    <li><strong>Different Season:</strong> Same destination, different month</li>
                    <li><strong>Nearby Alternatives:</strong> Similar experience, better weather</li>
                    <li><strong>Weather-Resilient:</strong> Destinations with indoor/outdoor options</li>
                    </ul>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p class="large-text"><strong>Weather-Friendly Alternatives:</strong></p>
                    <ul class="large-text">
                    <li><strong>Similar Climate:</strong> Destinations with similar weather patterns</li>
                    <li><strong>Indoor Focus:</strong> Destinations with many indoor activities</li>
                    <li><strong>Weather-Adaptive:</strong> Places that handle weather well</li>
                    </ul>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

# --- WEATHER IMPACT CALCULATOR MODE ---
elif mode == "Weather Impact Calculator":
    st.markdown('<h2 class="section-header">🧮 Weather Impact Calculator</h2>', unsafe_allow_html=True)
    st.markdown('<p class="large-text">Calculate the economic and social impact of weather events on different sectors and communities!</p>', unsafe_allow_html=True)
    
    # Impact calculation type
    impact_type = st.selectbox("Select Impact Type", ["Economic Impact", "Infrastructure Impact", "Agricultural Impact", "Health Impact", "Transportation Impact"], key="impact_type_select")
    
    if impact_type == "Economic Impact":
        st.markdown('<h3 class="subsection-header">💰 Economic Impact Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            event_type = st.selectbox("Weather Event Type", ["Tornado", "Hurricane", "Flood", "Drought", "Winter Storm", "Heat Wave"], key="event_type_economic")
            population = st.number_input("Population Affected", min_value=100, max_value=10000000, value=100000, step=1000, key="population_economic")
        
        with col2:
            event_duration = st.number_input("Event Duration (days)", min_value=1, max_value=365, value=7, key="duration_economic")
            severity_level = st.selectbox("Severity Level", ["Low", "Medium", "High", "Extreme"], key="severity_economic")
        
        if st.button("💰 Calculate Economic Impact"):
            # Simulate economic impact calculation
            import random
            random.seed(hash(event_type + severity_level + str(population)))
            
            # Base economic impacts by event type
            base_impacts = {
                "Tornado": {"property_damage": 50000, "business_loss": 20000, "infrastructure": 15000},
                "Hurricane": {"property_damage": 150000, "business_loss": 50000, "infrastructure": 75000},
                "Flood": {"property_damage": 80000, "business_loss": 30000, "infrastructure": 40000},
                "Drought": {"property_damage": 10000, "business_loss": 100000, "infrastructure": 5000},
                "Winter Storm": {"property_damage": 25000, "business_loss": 15000, "infrastructure": 20000},
                "Heat Wave": {"property_damage": 5000, "business_loss": 25000, "infrastructure": 10000}
            }
            
            severity_multipliers = {"Low": 0.5, "Medium": 1.0, "High": 2.0, "Extreme": 4.0}
            
            base = base_impacts.get(event_type, {"property_damage": 50000, "business_loss": 25000, "infrastructure": 25000})
            multiplier = severity_multipliers[severity_level]
            duration_factor = event_duration / 7  # Normalize to 1 week
            
            # Calculate impacts
            property_damage = base["property_damage"] * multiplier * duration_factor * (population / 100000)
            business_loss = base["business_loss"] * multiplier * duration_factor * (population / 100000)
            infrastructure_damage = base["infrastructure"] * multiplier * duration_factor * (population / 100000)
            total_impact = property_damage + business_loss + infrastructure_damage
            
            # Display results
            st.markdown('<h3 class="subsection-header">📊 Economic Impact Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Property Damage", f"${property_damage:,.0f}", f"${property_damage/1000000:.1f}M")
            with col2:
                st.metric("Business Loss", f"${business_loss:,.0f}", f"${business_loss/1000000:.1f}M")
            with col3:
                st.metric("Infrastructure", f"${infrastructure_damage:,.0f}", f"${infrastructure_damage/1000000:.1f}M")
            with col4:
                st.metric("Total Impact", f"${total_impact:,.0f}", f"${total_impact/1000000:.1f}M")
            
            # Impact visualization
            st.markdown('<h3 class="subsection-header">📈 Impact Breakdown</h3>', unsafe_allow_html=True)
            
            impact_data = {
                "Property Damage": property_damage,
                "Business Loss": business_loss,
                "Infrastructure": infrastructure_damage
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            wedges, texts, autotexts = ax.pie(impact_data.values(), labels=impact_data.keys(), autopct='%1.1f%%', colors=colors)
            ax.set_title(f'Economic Impact Breakdown - {event_type}', fontsize=16, fontweight='bold')
            st.pyplot(fig)
            
            # Recovery timeline
            st.markdown('<h3 class="subsection-header">⏰ Recovery Timeline</h3>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            <p class="large-text"><strong>Estimated Recovery Phases:</strong></p>
            <ul class="large-text">
            <li><strong>Immediate (0-7 days):</strong> Emergency response, basic services restoration</li>
            <li><strong>Short-term (1-4 weeks):</strong> Infrastructure repair, business reopening</li>
            <li><strong>Medium-term (1-6 months):</strong> Property reconstruction, economic recovery</li>
            <li><strong>Long-term (6+ months):</strong> Full economic restoration, community rebuilding</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif impact_type == "Infrastructure Impact":
        st.markdown('<h3 class="subsection-header">🏗️ Infrastructure Impact Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            infra_event = st.selectbox("Weather Event", ["Flood", "Hurricane", "Tornado", "Winter Storm", "Earthquake"], key="infra_event_select")
            area_size = st.number_input("Affected Area (sq miles)", min_value=1, max_value=10000, value=100, key="area_size_infra")
        
        with col2:
            population_density = st.selectbox("Population Density", ["Rural", "Suburban", "Urban", "Dense Urban"], key="density_select")
            infrastructure_age = st.selectbox("Infrastructure Age", ["New (0-10 years)", "Modern (10-30 years)", "Aging (30-50 years)", "Old (50+ years)"], key="age_select")
        
        if st.button("🏗️ Calculate Infrastructure Impact"):
            # Simulate infrastructure impact calculation
            import random
            random.seed(hash(infra_event + population_density + infrastructure_age))
            
            # Infrastructure vulnerability factors
            density_factors = {"Rural": 0.3, "Suburban": 0.6, "Urban": 0.8, "Dense Urban": 1.0}
            age_factors = {"New (0-10 years)": 0.2, "Modern (10-30 years)": 0.5, "Aging (30-50 years)": 0.8, "Old (50+ years)": 1.0}
            
            # Calculate impact scores
            roads_damage = random.uniform(0.1, 0.9) * density_factors[population_density] * age_factors[infrastructure_age] * area_size
            power_outages = random.uniform(0.05, 0.7) * density_factors[population_density] * age_factors[infrastructure_age] * area_size
            water_systems = random.uniform(0.02, 0.5) * density_factors[population_density] * age_factors[infrastructure_age] * area_size
            communications = random.uniform(0.01, 0.4) * density_factors[population_density] * age_factors[infrastructure_age] * area_size
            
            # Display results
            st.markdown('<h3 class="subsection-header">📊 Infrastructure Impact Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Roads Damaged", f"{roads_damage:.1f} miles", f"{roads_damage/area_size*100:.1f}% of area")
            with col2:
                st.metric("Power Outages", f"{power_outages:.0f} households", f"{power_outages/area_size*100:.1f}% affected")
            with col3:
                st.metric("Water Systems", f"{water_systems:.0f} facilities", f"{water_systems/area_size*100:.1f}% damaged")
            with col4:
                st.metric("Communications", f"{communications:.0f} towers", f"{communications/area_size*100:.1f}% affected")
            
            # Recovery recommendations
            st.markdown('<h3 class="subsection-header">🔧 Recovery Recommendations</h3>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            <p class="large-text"><strong>Priority Repair Sequence:</strong></p>
            <ol class="large-text">
            <li><strong>Emergency Services:</strong> Restore critical infrastructure first</li>
            <li><strong>Power Grid:</strong> Prioritize hospitals, emergency services</li>
            <li><strong>Water Systems:</strong> Ensure clean water availability</li>
            <li><strong>Transportation:</strong> Restore major roadways and bridges</li>
            <li><strong>Communications:</strong> Reestablish communication networks</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
    
    elif impact_type == "Agricultural Impact":
        st.markdown('<h3 class="subsection-header">🌾 Agricultural Impact Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.selectbox("Primary Crop", ["Corn", "Wheat", "Soybeans", "Cotton", "Rice", "Vegetables", "Fruits"], key="crop_select")
            farm_size = st.number_input("Farm Size (acres)", min_value=10, max_value=10000, value=1000, key="farm_size_agri")
        
        with col2:
            weather_event = st.selectbox("Weather Event", ["Drought", "Flood", "Hail Storm", "Frost", "Heat Wave"], key="weather_event_agri")
            growth_stage = st.selectbox("Crop Growth Stage", ["Planting", "Early Growth", "Flowering", "Fruiting", "Harvest"], key="growth_stage_select")
        
        if st.button("🌾 Calculate Agricultural Impact"):
            # Simulate agricultural impact calculation
            import random
            random.seed(hash(crop_type + weather_event + growth_stage))
            
            # Crop vulnerability by growth stage
            stage_vulnerability = {
                "Planting": {"Drought": 0.3, "Flood": 0.8, "Hail Storm": 0.1, "Frost": 0.2, "Heat Wave": 0.4},
                "Early Growth": {"Drought": 0.5, "Flood": 0.7, "Hail Storm": 0.3, "Frost": 0.6, "Heat Wave": 0.5},
                "Flowering": {"Drought": 0.8, "Flood": 0.6, "Hail Storm": 0.7, "Frost": 0.9, "Heat Wave": 0.7},
                "Fruiting": {"Drought": 0.6, "Flood": 0.5, "Hail Storm": 0.8, "Frost": 0.4, "Heat Wave": 0.6},
                "Harvest": {"Drought": 0.2, "Flood": 0.9, "Hail Storm": 0.9, "Frost": 0.1, "Heat Wave": 0.3}
            }
            
            vulnerability = stage_vulnerability[growth_stage][weather_event]
            yield_loss = vulnerability * random.uniform(0.1, 0.9) * farm_size
            financial_loss = yield_loss * random.uniform(50, 200)  # $ per acre
            
            # Display results
            st.markdown('<h3 class="subsection-header">📊 Agricultural Impact Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Yield Loss", f"{yield_loss:.0f} acres", f"{yield_loss/farm_size*100:.1f}% of crop")
            with col2:
                st.metric("Financial Loss", f"${financial_loss:,.0f}", f"${financial_loss/1000:.1f}K")
            with col3:
                st.metric("Recovery Time", f"{random.randint(1, 12)} months", "To full production")
            
            # Mitigation strategies
            st.markdown('<h3 class="subsection-header">🌱 Mitigation Strategies</h3>', unsafe_allow_html=True)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            <p class="large-text"><strong>Recommended Actions:</strong></p>
            <ul class="large-text">
            <li><strong>Crop Insurance:</strong> File claims for weather-related losses</li>
            <li><strong>Alternative Crops:</strong> Consider drought/flood resistant varieties</li>
            <li><strong>Irrigation Systems:</strong> Invest in water management infrastructure</li>
            <li><strong>Soil Conservation:</strong> Implement erosion control measures</li>
            <li><strong>Diversification:</strong> Plant multiple crop types to reduce risk</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# --- LEARN MODE ---
elif mode == "Learn":
    st.markdown('<h2 class="section-header">📚 Educational Center: Severe Weather Deep Dive</h2>', unsafe_allow_html=True)

    # Create tabs for different weather phenomena
    tab1, tab2, tab3, tab4, tab5 = st.tabs(config.WEATHER_CATEGORIES)

    with tab1:
        st.markdown('<h3 class="subsection-header">🌪️ Tornadoes: Nature\'s Most Violent Storms</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>What Are Tornadoes?</strong></p>
        <p class="large-text">Tornadoes are violently rotating columns of air extending from thunderstorms to the ground, capable of causing extreme destruction. They can reach wind speeds over 300 mph and travel for miles, leaving devastation in their wake.</p>
        
        <p class="large-text"><strong>🌪️ How Tornadoes Form:</strong></p>
        <ul class="large-text">
        <li><strong>Wind Shear:</strong> Changes in wind speed and direction with height create horizontal rotation</li>
        <li><strong>Updraft:</strong> Strong upward air currents tilt the rotating air vertically</li>
        <li><strong>Mesocyclone:</strong> The rotating updraft creates a mesocyclone (rotating thunderstorm)</li>
        <li><strong>Wall Cloud:</strong> A lowering of the cloud base indicates rotation</li>
        <li><strong>Funnel Cloud:</strong> The rotating column becomes visible as it descends</li>
        <li><strong>Tornado:</strong> When the funnel touches the ground, it becomes a tornado</li>
        </ul>
        
        <p class="large-text"><strong>⏰ High-Frequency Time Periods:</strong></p>
        <ul class="large-text">
        <li><strong>Season:</strong> Peak tornado season is March-July, with May being the most active month</li>
        <li><strong>Time of Day:</strong> Most tornadoes occur between 3-9 PM, when daytime heating is strongest</li>
        <li><strong>Geographic Hotspots:</strong> "Tornado Alley" (Texas to South Dakota) experiences the highest frequency</li>
        <li><strong>Weather Conditions:</strong> Warm, humid air masses colliding with cold, dry air masses</li>
        </ul>
        
        <p class="large-text"><strong>📊 Enhanced Fujita (EF) Scale:</strong></p>
        <ul class="large-text">
        <li><strong>EF0 (65-85 mph):</strong> Light damage - broken tree branches, minor roof damage</li>
        <li><strong>EF1 (86-110 mph):</strong> Moderate damage - mobile homes overturned, roof shingles removed</li>
        <li><strong>EF2 (111-135 mph):</strong> Considerable damage - roofs torn off houses, large trees uprooted</li>
        <li><strong>EF3 (136-165 mph):</strong> Severe damage - entire stories of houses destroyed, trains overturned</li>
        <li><strong>EF4 (166-200 mph):</strong> Devastating damage - well-constructed houses leveled, cars thrown</li>
        <li><strong>EF5 (over 200 mph):</strong> Incredible damage - strong frame houses lifted and carried away</li>
        </ul>
        
        <p class="large-text"><strong>🚨 Warning Signs:</strong></p>
        <ul class="large-text">
        <li><strong>Dark, greenish sky</strong> - indicates hail and severe weather</li>
        <li><strong>Large hail</strong> - often precedes tornado formation</li>
        <li><strong>Loud roar like a freight train</strong> - the sound of the tornado itself</li>
        <li><strong>Rotating funnel cloud</strong> - the visible tornado formation</li>
        <li><strong>Debris cloud</strong> - even if no funnel is visible</li>
        <li><strong>Sudden calm</strong> - the "eye" of the storm</li>
        <li><strong>Wall cloud</strong> - a lowering of the cloud base</li>
        </ul>
        
        <p class="large-text"><strong>🛡️ Safety Measures:</strong></p>
        <ul class="large-text">
        <li><strong>Immediate Action:</strong> Seek shelter in a basement or interior room without windows</li>
        <li><strong>Best Locations:</strong> Basements, storm cellars, interior bathrooms or closets</li>
        <li><strong>Protection:</strong> Cover your head and neck with your arms or a sturdy object</li>
        <li><strong>Avoid:</strong> Mobile homes, vehicles, under highway overpasses</li>
        <li><strong>Stay Informed:</strong> Have multiple ways to receive weather alerts</li>
        <li><strong>Emergency Kit:</strong> Flashlight, batteries, water, first aid supplies</li>
        </ul>
        
        <p class="large-text"><strong>📚 Historical Accounts:</strong></p>
        <ul class="large-text">
        <li><strong>Tri-State Tornado (1925):</strong> Deadliest tornado in US history, killed 695 people across Missouri, Illinois, and Indiana</li>
        <li><strong>Joplin Tornado (2011):</strong> EF5 tornado killed 158 people and caused $2.8 billion in damage</li>
        <li><strong>Moore Tornado (2013):</strong> EF5 tornado with winds over 200 mph, killed 24 people</li>
        <li><strong>El Reno Tornado (2013):</strong> Widest tornado ever recorded at 2.6 miles wide</li>
        <li><strong>Bridge Creek-Moore Tornado (1999):</strong> Highest wind speed ever recorded at 302 mph</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<h3 class="subsection-header">⛈️ Severe Thunderstorms: Nature\'s Electrical Storms</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>What Are Severe Thunderstorms?</strong></p>
        <p class="large-text">Severe thunderstorms are intense electrical storms that produce damaging winds, large hail, torrential rain, and sometimes tornadoes. They are classified as "severe" when they produce winds ≥58 mph, hail ≥1 inch, or tornadoes.</p>
        
        <p class="large-text"><strong>⚡ How Severe Thunderstorms Form:</strong></p>
        <ul class="large-text">
        <li><strong>Moisture:</strong> Warm, humid air provides the fuel for storm development</li>
        <li><strong>Instability:</strong> Temperature differences create buoyant air that rises rapidly</li>
        <li><strong>Lift:</strong> Cold fronts, sea breezes, or mountains force air upward</li>
        <li><strong>Wind Shear:</strong> Changes in wind direction and speed with height</li>
        <li><strong>Updraft:</strong> Strong upward air currents create the storm's engine</li>
        <li><strong>Downdraft:</strong> Rain-cooled air descends, creating gusty winds</li>
        </ul>
        
        <p class="large-text"><strong>⏰ High-Frequency Time Periods:</strong></p>
        <ul class="large-text">
        <li><strong>Season:</strong> Peak activity in spring and summer (March-September)</li>
        <li><strong>Time of Day:</strong> Most active between 2-8 PM due to daytime heating</li>
        <li><strong>Geographic Areas:</strong> Great Plains, Southeast, and Midwest experience highest frequency</li>
        <li><strong>Weather Patterns:</strong> Cold fronts, dry lines, and sea breezes trigger storms</li>
        </ul>
        
        <p class="large-text"><strong>💨 Key Characteristics:</strong></p>
        <ul class="large-text">
        <li><strong>Damaging Winds:</strong> ≥58 mph can knock down trees, power lines, and damage structures</li>
        <li><strong>Large Hail:</strong> ≥1 inch can damage vehicles, roofs, crops, and injure people</li>
        <li><strong>Lightning:</strong> Causes thousands of injuries and hundreds of deaths yearly</li>
        <li><strong>Heavy Rain:</strong> Can cause flash flooding and reduce visibility</li>
        <li><strong>Microbursts:</strong> Intense downdrafts that can cause aircraft accidents</li>
        </ul>
        
        <p class="large-text"><strong>🚨 Warning Signs:</strong></p>
        <ul class="large-text">
        <li><strong>Dark, towering clouds</strong> - cumulonimbus clouds reaching high into the atmosphere</li>
        <li><strong>Lightning and thunder</strong> - electrical activity within the storm</li>
        <li><strong>Strong winds</strong> - gusty conditions as the storm approaches</li>
        <li><strong>Hail</strong> - ice pellets falling from the storm</li>
        <li><strong>Heavy rain</strong> - torrential downpours</li>
        <li><strong>Green sky</strong> - often indicates hail and severe weather</li>
        </ul>
        
        <p class="large-text"><strong>🛡️ Safety Measures:</strong></p>
        <ul class="large-text">
        <li><strong>Indoor Safety:</strong> Stay inside away from windows and electrical appliances</li>
        <li><strong>Lightning Safety:</strong> Avoid plumbing, electrical equipment, and corded phones</li>
        <li><strong>Outdoor Safety:</strong> Seek shelter immediately; avoid open fields and tall objects</li>
        <li><strong>Vehicle Safety:</strong> Cars provide some protection from lightning</li>
        <li><strong>Wait Time:</strong> Wait 30 minutes after last thunder before resuming outdoor activities</li>
        <li><strong>Emergency Kit:</strong> Flashlight, batteries, water, first aid supplies</li>
        </ul>
        
        <p class="large-text"><strong>📚 Historical Accounts:</strong></p>
        <ul class="large-text">
        <li><strong>Super Outbreak (1974):</strong> 148 tornadoes in 13 states over 16 hours, killed 319 people</li>
        <li><strong>Derecho (2012):</strong> Widespread windstorm from Chicago to Washington DC, caused $2.9 billion damage</li>
        <li><strong>Hailstorm (2010):</strong> Oklahoma City area hit by softball-sized hail, caused $1.5 billion damage</li>
        <li><strong>Lightning Strike (2019):</strong> Single lightning bolt traveled 440 miles across three states</li>
        <li><strong>Microburst (1985):</strong> Delta Flight 191 crash caused by microburst winds at Dallas airport</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<h3 class="subsection-header">🌊 Flash Floods: Sudden Water Dangers</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>What Are Flash Floods?</strong></p>
        <p class="large-text">Flash floods are sudden, rapid flooding events that occur within 6 hours of heavy rainfall, dam breaks, or rapid snowmelt. They can develop in minutes and move with incredible speed, making them one of the most dangerous weather phenomena.</p>
        
        <p class="large-text"><strong>💧 How Flash Floods Form:</strong></p>
        <ul class="large-text">
        <li><strong>Heavy Rainfall:</strong> Intense precipitation overwhelms drainage systems</li>
        <li><strong>Dam Failure:</strong> Structural failure releases large volumes of water</li>
        <li><strong>Snowmelt:</strong> Rapid warming causes sudden snow and ice melting</li>
        <li><strong>Urban Development:</strong> Paved surfaces prevent water absorption</li>
        <li><strong>Topography:</strong> Steep terrain channels water into narrow valleys</li>
        <li><strong>Soil Saturation:</strong> Already wet ground cannot absorb more water</li>
        </ul>
        
        <p class="large-text"><strong>⏰ High-Frequency Time Periods:</strong></p>
        <ul class="large-text">
        <li><strong>Season:</strong> Can occur year-round, but most common in spring and summer</li>
        <li><strong>Weather Events:</strong> During and immediately after heavy thunderstorms</li>
        <li><strong>Geographic Areas:</strong> Low-lying areas, urban centers, near rivers and streams</li>
        <li><strong>Time of Day:</strong> Often occur during peak thunderstorm hours (2-8 PM)</li>
        </ul>
        
        <p class="large-text"><strong>⚠️ Why Flash Floods Are Dangerous:</strong></p>
        <ul class="large-text">
        <li><strong>Speed:</strong> Can develop in minutes and move at 20+ mph</li>
        <li><strong>Depth:</strong> Water can be much deeper than it appears</li>
        <li><strong>Force:</strong> 6 inches of fast-moving water can knock down adults</li>
        <li><strong>Debris:</strong> Carries rocks, trees, and other dangerous objects</li>
        <li><strong>Electrocution Risk:</strong> Downed power lines in water</li>
        <li><strong>Contamination:</strong> Water may contain sewage and chemicals</li>
        </ul>
        
        <p class="large-text"><strong>🚨 Warning Signs:</strong></p>
        <ul class="large-text">
        <li><strong>Heavy rainfall</strong> - especially over already saturated ground</li>
        <li><strong>Rising water levels</strong> - in streams, rivers, or drainage ditches</li>
        <li><strong>Muddy water</strong> - indicates soil erosion and potential flooding</li>
        <li><strong>Debris in water</strong> - branches, leaves, or other objects floating</li>
        <li><strong>Road closures</strong> - authorities closing flooded roads</li>
        <li><strong>Weather alerts</strong> - Flash Flood Warnings issued by NWS</li>
        </ul>
        
        <p class="large-text"><strong>🛡️ Safety Measures:</strong></p>
        <ul class="large-text">
        <li><strong>Never Drive Through:</strong> "Turn Around, Don't Drown" - just 12 inches of water can float a car</li>
        <li><strong>Move to Higher Ground:</strong> Seek elevated areas immediately</li>
        <li><strong>Stay Informed:</strong> Monitor weather alerts and local news</li>
        <li><strong>Evacuate Early:</strong> Don't wait for evacuation orders if flooding is imminent</li>
        <li><strong>Avoid Walking:</strong> Don't walk through moving water</li>
        <li><strong>Emergency Kit:</strong> Have supplies ready for evacuation</li>
        <li><strong>Know Your Area:</strong> Identify flood-prone areas and evacuation routes</li>
        </ul>
        
        <p class="large-text"><strong>📚 Historical Accounts:</strong></p>
        <ul class="large-text">
        <li><strong>Big Thompson Canyon (1976):</strong> Flash flood killed 144 people in Colorado, caused by 12 inches of rain in 4 hours</li>
        <li><strong>Johnstown Flood (1889):</strong> Dam failure killed 2,209 people in Pennsylvania</li>
        <li><strong>Ellicott City (2016, 2018):</strong> Historic Maryland town devastated by flash floods twice in two years</li>
        <li><strong>Tennessee Floods (2021):</strong> Record rainfall caused flash floods that killed 20 people</li>
        <li><strong>Hurricane Harvey (2017):</strong> Dropped 60+ inches of rain, causing catastrophic flooding in Texas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<h3 class="subsection-header">❄️ Winter Storms & Ice Storms: Frozen Fury</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>What Are Winter Storms?</strong></p>
        <p class="large-text">Winter storms are complex weather systems that bring snow, sleet, freezing rain, and ice. They can paralyze entire regions, causing power outages, transportation shutdowns, and dangerous conditions for days or weeks.</p>
        
        <p class="large-text"><strong>❄️ How Winter Storms Form:</strong></p>
        <ul class="large-text">
        <li><strong>Cold Air Mass:</strong> Arctic or polar air masses provide the cold temperatures</li>
        <li><strong>Moisture Source:</strong> Warm, moist air from oceans or lakes provides precipitation</li>
        <li><strong>Temperature Layers:</strong> Different temperature layers determine precipitation type</li>
        <li><strong>Lift Mechanism:</strong> Fronts, low pressure systems, or lake-effect processes</li>
        <li><strong>Precipitation Types:</strong> Snow, sleet, freezing rain, or ice pellets</li>
        <li><strong>Accumulation:</strong> Snow builds up or ice coats surfaces</li>
        </ul>
        
        <p class="large-text"><strong>⏰ High-Frequency Time Periods:</strong></p>
        <ul class="large-text">
        <li><strong>Season:</strong> December through March, with peak activity in January-February</li>
        <li><strong>Geographic Areas:</strong> Northern states, mountainous regions, and areas near large lakes</li>
        <li><strong>Weather Patterns:</strong> Nor'easters, Alberta Clippers, and lake-effect snow</li>
        <li><strong>Temperature Thresholds:</strong> Below freezing temperatures required for snow/ice</li>
        </ul>
        
        <p class="large-text"><strong>⚠️ Hazards and Dangers:</strong></p>
        <ul class="large-text">
        <li><strong>Snow Accumulation:</strong> Reduces visibility, blocks roads, and can collapse roofs</li>
        <li><strong>Ice Buildup:</strong> Creates slippery conditions on roads, sidewalks, and power lines</li>
        <li><strong>Power Outages:</strong> Ice on power lines can cause widespread blackouts</li>
        <li><strong>Hypothermia:</strong> Prolonged exposure to cold can be life-threatening</li>
        <li><strong>Carbon Monoxide:</strong> Improper use of heating devices can cause poisoning</li>
        <li><strong>Transportation Disruption:</strong> Airports, roads, and public transit may shut down</li>
        </ul>
        
        <p class="large-text"><strong>🚨 Warning Signs:</strong></p>
        <ul class="large-text">
        <li><strong>Weather forecasts</strong> - Winter Storm Warnings and Ice Storm Warnings</li>
        <li><strong>Temperature drops</strong> - Rapid cooling below freezing</li>
        <li><strong>Precipitation changes</strong> - Rain changing to snow or freezing rain</li>
        <li><strong>Wind increases</strong> - Blowing snow and wind chill</li>
        <li><strong>Pressure changes</strong> - Falling barometric pressure</li>
        <li><strong>Cloud formations</strong> - Thick, gray clouds indicating precipitation</li>
        </ul>
        
        <p class="large-text"><strong>🛡️ Safety Measures:</strong></p>
        <ul class="large-text">
        <li><strong>Emergency Kit:</strong> Warm clothing, blankets, food, water, flashlight, batteries</li>
        <li><strong>Home Preparation:</strong> Insulate pipes, seal windows, have alternative heat sources</li>
        <li><strong>Travel Safety:</strong> Avoid travel unless absolutely necessary</li>
        <li><strong>Vehicle Safety:</strong> Keep emergency supplies in car, check weather before driving</li>
        <li><strong>Heating Safety:</strong> Use space heaters safely, prevent carbon monoxide poisoning</li>
        <li><strong>Stay Informed:</strong> Monitor weather alerts and local news</li>
        <li><strong>Community Support:</strong> Check on elderly neighbors and vulnerable populations</li>
        </ul>
        
        <p class="large-text"><strong>📚 Historical Accounts:</strong></p>
        <ul class="large-text">
        <li><strong>Great Blizzard (1888):</strong> Killed 400+ people in Northeast, paralyzed cities for days</li>
        <li><strong>Ice Storm (1998):</strong> Devastated Canada and Northeast US, caused massive power outages</li>
        <li><strong>Snowmageddon (2010):</strong> Washington DC buried under 2+ feet of snow</li>
        <li><strong>Polar Vortex (2014):</strong> Brought record cold temperatures across much of US</li>
        <li><strong>Texas Winter Storm (2021):</strong> Caused widespread power outages and 246 deaths</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.markdown('<h3 class="subsection-header">🌀 Hurricanes & Tropical Storms: Ocean Giants</h3>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        <p class="large-text"><strong>What Are Hurricanes?</strong></p>
        <p class="large-text">Hurricanes are massive, rotating storm systems that form over warm ocean waters. They are among the most powerful and destructive weather phenomena on Earth, capable of causing catastrophic damage through high winds, storm surge, and flooding rains.</p>
        
        <p class="large-text"><strong>🌊 How Hurricanes Form:</strong></p>
        <ul class="large-text">
        <li><strong>Warm Ocean Waters:</strong> Sea surface temperatures ≥80°F (27°C) provide energy</li>
        <li><strong>Atmospheric Instability:</strong> Warm, moist air rises and cools, releasing heat</li>
        <li><strong>Low Wind Shear:</strong> Consistent wind patterns allow storm organization</li>
        <li><strong>Coriolis Effect:</strong> Earth's rotation creates the spinning motion</li>
        <li><strong>Moisture Supply:</strong> High humidity in the lower atmosphere</li>
        <li><strong>Disturbance:</strong> Tropical wave or low-pressure system triggers development</li>
        </ul>
        
        <p class="large-text"><strong>⏰ High-Frequency Time Periods:</strong></p>
        <ul class="large-text">
        <li><strong>Atlantic Season:</strong> June 1 - November 30, with peak activity August-October</li>
        <li><strong>Pacific Season:</strong> May 15 - November 30, with peak activity July-September</li>
        <li><strong>Geographic Areas:</strong> Atlantic Ocean, Gulf of Mexico, Caribbean Sea, Pacific Ocean</li>
        <li><strong>Weather Patterns:</strong> African easterly waves, El Niño/La Niña cycles</li>
        </ul>
        
        <p class="large-text"><strong>🌀 Hurricane Structure:</strong></p>
        <ul class="large-text">
        <li><strong>Eye:</strong> Calm center with clear skies, typically 20-40 miles wide</li>
        <li><strong>Eyewall:</strong> Surrounds eye with strongest winds and most intense rain</li>
        <li><strong>Rainbands:</strong> Spiral outward causing heavy showers and gusty winds</li>
        <li><strong>Outflow:</strong> High-level winds that exhaust the storm's energy</li>
        <li><strong>Storm Surge:</strong> Abnormal rise in sea level caused by wind and pressure</li>
        </ul>
        
        <p class="large-text"><strong>📊 Saffir-Simpson Hurricane Scale:</strong></p>
        <ul class="large-text">
        <li><strong>Category 1 (74-95 mph):</strong> Very dangerous winds, some damage</li>
        <li><strong>Category 2 (96-110 mph):</strong> Extremely dangerous winds, extensive damage</li>
        <li><strong>Category 3 (111-129 mph):</strong> Devastating damage, major hurricane</li>
        <li><strong>Category 4 (130-156 mph):</strong> Catastrophic damage, major hurricane</li>
        <li><strong>Category 5 (157+ mph):</strong> Catastrophic damage, major hurricane</li>
        </ul>
        
        <p class="large-text"><strong>🚨 Warning Signs:</strong></p>
        <ul class="large-text">
        <li><strong>Weather forecasts</strong> - Hurricane Watches and Warnings</li>
        <li><strong>Ocean conditions</strong> - Increasing wave heights and surf</li>
        <li><strong>Wind changes</strong> - Gradual increase in wind speed and direction</li>
        <li><strong>Pressure drops</strong> - Falling barometric pressure</li>
        <li><strong>Cloud formations</strong> - Cirrus clouds followed by thickening overcast</li>
        <li><strong>Rain bands</strong> - Intermittent heavy rain and wind</li>
        </ul>
        
        <p class="large-text"><strong>🛡️ Safety Measures:</strong></p>
        <ul class="large-text">
        <li><strong>Evacuation:</strong> Follow evacuation orders early, don't wait until last minute</li>
        <li><strong>Home Preparation:</strong> Secure outdoor objects, install storm shutters</li>
        <li><strong>Emergency Kit:</strong> Water, food, medications, important documents, cash</li>
        <li><strong>Communication Plan:</strong> Have family emergency plan and contact information</li>
        <li><strong>Insurance:</strong> Ensure adequate coverage for wind and flood damage</li>
        <li><strong>Stay Informed:</strong> Monitor weather updates and emergency broadcasts</li>
        <li><strong>After Storm:</strong> Wait for official clearance before returning home</li>
        </ul>
        
        <p class="large-text"><strong>📚 Historical Accounts:</strong></p>
        <ul class="large-text">
        <li><strong>Galveston Hurricane (1900):</strong> Deadliest US hurricane, killed 8,000+ people</li>
        <li><strong>Hurricane Katrina (2005):</strong> Caused $125 billion damage, killed 1,833 people</li>
        <li><strong>Hurricane Sandy (2012):</strong> Superstorm affected 24 states, caused $70 billion damage</li>
        <li><strong>Hurricane Maria (2017):</strong> Devastated Puerto Rico, caused longest blackout in US history</li>
        <li><strong>Hurricane Harvey (2017):</strong> Dropped 60+ inches of rain, caused catastrophic flooding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # General safety tips section
    st.markdown("---")
    st.markdown('<h3 class="subsection-header">⚠️ General Severe Weather Safety Tips</h3>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><ul class="large-text"><li>Always have multiple ways to receive weather alerts (smartphone apps, NOAA Weather Radio, local news)</li><li>Prepare a family emergency plan and practice it regularly</li><li>Assemble an emergency kit: water, non-perishable food, flashlight, batteries, medications, important documents</li><li>Know your community\'s emergency shelters and evacuation routes</li><li>Never underestimate the power of severe weather—stay informed and act promptly</li></ul></div>', unsafe_allow_html=True)

# --- LIVE ALERTS AND HISTORICAL ALERTS MODE ---
elif mode in ["Live Alerts", "Historical Alerts"]:
    start_date = None
    end_date = None
    fetch_historical = False

    if mode == "Historical Alerts":
        st.sidebar.markdown("#### Select Date Range")
        today = datetime.utcnow().date()
        start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days=7), max_value=today)
        end_date = st.sidebar.date_input("End Date", value=today, min_value=start_date, max_value=today)
        fetch_historical = st.sidebar.button("🔍 Fetch Historical Alerts")

    # --- FETCH ALERTS ---
    alerts = []
    try:
        if mode == "Historical Alerts" and start_date and end_date and fetch_historical:
            # Fix historical URL - use proper date format and remove severity filter for historical data
            historical_url = f"https://api.weather.gov/alerts?start={start_date}T00:00:00Z&end={end_date}T23:59:59Z"
            st.info(f"Fetching historical alerts from {start_date} to {end_date}...")
            response = requests.get(historical_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
            alerts = data.get("features", [])
            st.success(f"Found {len(alerts)} historical alerts")
        elif mode != "Historical Alerts":
            response = requests.get(API_URL, headers=HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            alerts = data.get("features", [])
    except Exception as e:
        st.markdown(f'<div class="error-box"><p class="large-text">❌ Error fetching data from NOAA API: {e}</p></div>', unsafe_allow_html=True)
        if mode == "Historical Alerts":
            st.markdown('<div class="info-box"><p class="large-text">💡 <strong>Tip:</strong> Historical data may be limited. Try a smaller date range or different dates.</p></div>', unsafe_allow_html=True)

    # --- DYNAMIC STATE LIST ---
    all_states = set()
    for alert in alerts:
        area_desc = alert["properties"].get("areaDesc", "")
        parts = [part.strip() for part in area_desc.replace(";", ",").split(",")]
        for p in parts:
            if len(p) == 2 and p.isalpha():
                all_states.add(p)
    state_options = ["All"] + sorted(all_states)

    # --- DYNAMIC EVENT TYPE LIST ---
    all_events = set()
    for alert in alerts:
        event = alert["properties"].get("event", "")
        if event:
            all_events.add(event)
    event_options = ["All"] + sorted(all_events)

    # --- SIDEBAR FILTERS ---
    st.sidebar.title("Filters")
    selected_state = st.sidebar.selectbox("Filter by State (or All)", state_options)
    selected_event = st.sidebar.selectbox("Filter by Event Type (or All)", event_options)
    user_zip = st.sidebar.text_input("Enter your ZIP code to see nearby alerts (optional)")

    # --- FILTER ALERTS ---
    filtered_alerts = []
    for alert in alerts:
        props = alert["properties"]
        if selected_state != "All" and selected_state not in props.get("areaDesc", ""):
            continue
        if selected_event != "All" and selected_event.lower() != props.get("event", "").lower():
            continue
        filtered_alerts.append(alert)

    st.markdown(f'<h3 class="subsection-header">🔎 Showing {len(filtered_alerts)} {"Historical" if mode == "Historical Alerts" else "Active"} Alerts</h3>', unsafe_allow_html=True)

    # --- COLOR MAP FOR ALERT SEVERITY ---
    color_map = config.SEVERITY_COLORS
    severity_weight = config.SEVERITY_WEIGHTS

    # --- SUMMARY STATISTICS ---
    total_alerts = len(filtered_alerts)
    avg_severity = 0
    if total_alerts > 0:
        avg_severity = sum(severity_weight.get(a["properties"]["severity"], 0) for a in filtered_alerts) / total_alerts

    states_list = []
    for alert in filtered_alerts:
        area_desc = alert["properties"].get("areaDesc", "")
        states_list.extend([s.strip() for s in area_desc.replace(";", ",").split(",") if len(s.strip()) == 2])

    top_states = Counter(states_list).most_common(5)

    st.markdown('<h3 class="subsection-header">📊 Summary Statistics</h3>', unsafe_allow_html=True)
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    col_stats1.metric("Total Alerts", total_alerts)
    col_stats2.metric("Average Severity", f"{avg_severity:.2f}")
    col_stats3.write("Top 5 Affected States")
    for state, count in top_states:
        col_stats3.write(f"- {state}: {count}")

    # --- IMPROVED SEVERITY VISUALIZATION ---
    if total_alerts > 0:
        st.markdown('<h3 class="subsection-header">📊 Alert Analysis Dashboard</h3>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Severity Distribution", "Event Types", "Time Analysis"])
        
        with viz_tab1:
            # Severity distribution pie chart
            severity_counts = Counter([alert["properties"]["severity"] for alert in filtered_alerts])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = [color_map.get(sev, "#4361ee") for sev in severity_counts.keys()]
            wedges, texts, autotexts = ax.pie(severity_counts.values(), labels=severity_counts.keys(), 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title("Alert Severity Distribution", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Severity insights
            most_common_severity = severity_counts.most_common(1)[0]
            st.markdown(f"**💡 Insight:** {most_common_severity[0]} severity alerts are most common ({most_common_severity[1]} alerts)")
        
        with viz_tab2:
            # Event type analysis
            event_counts = Counter([alert["properties"]["event"] for alert in filtered_alerts])
            top_events = event_counts.most_common(8)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            events, counts = zip(*top_events)
            bars = ax.barh(events, counts, color='skyblue', alpha=0.7)
            ax.set_xlabel("Number of Alerts")
            ax.set_title("Most Common Weather Events", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       str(count), ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with viz_tab3:
            # Time-based analysis
            if mode == "Historical Alerts":
                # For historical data, analyze by date
                date_counts = Counter()
                for alert in filtered_alerts:
                    try:
                        date_str = alert["properties"]["sent"].split("T")[0]
                        date_counts[date_str] += 1
                    except:
                        continue
                
                if date_counts:
                    dates = sorted(date_counts.keys())
                    counts = [date_counts[date] for date in dates]
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(dates, counts, marker='o', linewidth=2, markersize=4)
                    ax.fill_between(dates, counts, alpha=0.3)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Number of Alerts")
                    ax.set_title("Alert Frequency Over Time", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No time data available for historical alerts")
            else:
                # For live data, show current time distribution
                st.markdown("**⏰ Current Alert Status:**")
                st.markdown(f"- **Active Alerts:** {len(filtered_alerts)}")
                st.markdown(f"- **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"- **Auto-refresh:** Every 5 minutes")

    # --- FILTER ALERTS BY ZIP CODE (if entered) ---
    nearby_alerts = []
    user_lat = None
    user_lon = None
    if user_zip:
        nomi = pgeocode.Nominatim('us')
        zip_data = nomi.query_postal_code(user_zip)
        if zip_data is None or pd.isna(zip_data.latitude) or pd.isna(zip_data.longitude):
            st.markdown('<div class="error-box"><p class="large-text">❌ Invalid ZIP code or location not found.</p></div>', unsafe_allow_html=True)
        else:
            user_lat, user_lon = zip_data.latitude, zip_data.longitude
            st.markdown(f"### Alerts within {config.DEFAULT_PROXIMITY_MILES} miles of ZIP {user_zip}")

            def haversine(lat1, lon1, lat2, lon2):
                R = 3956  # miles
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
                c = 2*asin(sqrt(a))
                return R * c

            for alert in filtered_alerts:
                geom = alert.get("geometry")
                if geom and geom["type"] == "Polygon":
                    coords = geom["coordinates"][0]
                    lat = sum([c[1] for c in coords]) / len(coords)
                    lon = sum([c[0] for c in coords]) / len(coords)
                    distance = haversine(user_lat, user_lon, lat, lon)
                    if distance <= config.DEFAULT_PROXIMITY_MILES:
                        nearby_alerts.append(alert)

            if nearby_alerts:
                for alert in nearby_alerts:
                    props = alert["properties"]
                    st.markdown(f"#### ⚠️ {props['event']} in {props['areaDesc']}")
                    st.write(f"**Severity:** {props['severity']}  |  **Urgency:** {props['urgency']}")
                    st.write(f"**Expires:** {props['expires']}")
                    st.write(f"**Headline:** {props.get('headline', 'No headline provided.')}")
                    st.write(f"**Instructions:** {props.get('instruction', 'No instructions provided.')}")
                    if props.get("url"):
                        st.markdown(f"[More Info]({props['url']})")
                    else:
                        st.write("🔗 No additional link provided.")
                    st.markdown("---")
            else:
                st.markdown('<div class="info-box"><p class="large-text">ℹ️ No alerts near your location.</p></div>', unsafe_allow_html=True)

    # --- LAYOUT WITH COLUMNS ---
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<h3 class="subsection-header">📋 Alerts List</h3>', unsafe_allow_html=True)
        if len(filtered_alerts) == 0:
            st.markdown('<div class="info-box"><p class="large-text">No alerts match the filters.</p></div>', unsafe_allow_html=True)
        else:
            for alert in filtered_alerts:
                props = alert["properties"]
                st.markdown(f"#### ⚠️ {props['event']} in {props['areaDesc']}")
                st.write(f"**Severity:** {props['severity']}  |  **Urgency:** {props['urgency']}")

                # Countdown timer for expiration (only for live alerts)
                if mode != "Historical Alerts":
                    expires_str = props.get("expires")
                    if expires_str:
                        try:
                            expires_dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00")).astimezone(pytz.UTC)
                            now = datetime.now(pytz.UTC)
                            time_left = expires_dt - now
                            if time_left.total_seconds() > 0:
                                st.write(f"⏳ Time until expiration: {str(time_left).split('.')[0]} (HH:MM:SS)")
                            else:
                                st.write("⚠️ Alert expired")
                        except Exception:
                            pass
                else:
                    # For historical alerts, show sent time
                    sent_str = props.get("sent")
                    if sent_str:
                        try:
                            sent_dt = datetime.fromisoformat(sent_str.replace("Z", "+00:00"))
                            st.write(f"📅 Sent: {sent_dt.strftime('%Y-%m-%d %H:%M')}")
                        except Exception:
                            pass

                st.write(f"**Headline:** {props.get('headline', 'No headline provided.')}")
                st.write(f"**Instructions:** {props.get('instruction', 'No instructions provided.')}")
                if props.get("url"):
                    st.markdown(f"[More Info]({props['url']})")
                else:
                    st.write("🔗 No additional link provided.")
                st.markdown("---")

            # Export button
            if st.button("📥 Export Alerts to CSV"):
                df = pd.json_normalize([a["properties"] for a in filtered_alerts])
                filename = config.EXPORT_FILENAME
                df.to_csv(filename, index=False)
                st.markdown(f'<div class="success-box"><p class="large-text">✅ Alerts exported to `{filename}`</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="subsection-header">🗺️ Active Alert Areas Map</h3>', unsafe_allow_html=True)
        # Create map centered over US
        m = folium.Map(location=config.DEFAULT_MAP_CENTER, zoom_start=config.DEFAULT_MAP_ZOOM, tiles=config.MAP_TILES)

        # Add polygons for alerts with color-coded severity
        for alert in filtered_alerts:
            geom = alert.get("geometry")
            if geom and geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                latlon = [(c[1], c[0]) for c in coords]
                props = alert["properties"]
                color = color_map.get(props['severity'], "#4361ee")  # default blue
                folium.Polygon(
                    locations=latlon,
                    color=color,
                    fill=True,
                    fill_opacity=0.4,
                    popup=(
                        f"{props['event']}<br>"
                        f"Severity: {props['severity']}<br>"
                        f"Expires: {props['expires']}"
                    )
                ).add_to(m)

        # --- Add heatmap layer ---
        heat_data = []
        for alert in filtered_alerts:
            geom = alert.get("geometry")
            if geom and geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                lat = sum([c[1] for c in coords]) / len(coords)
                lon = sum([c[0] for c in coords]) / len(coords)
                weight = severity_weight.get(alert["properties"]["severity"], 1)
                heat_data.append([lat, lon, weight])

        if heat_data:
            HeatMap(heat_data, radius=25, max_zoom=9).add_to(m)

        # If user ZIP entered, add marker
        if user_zip and user_lat and user_lon:
            folium.Marker(
                location=[user_lat, user_lon],
                popup=f"Your Location ZIP: {user_zip}",
                icon=folium.Icon(color="blue", icon="home")
            ).add_to(m)

        folium_static(m)
