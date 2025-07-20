"""
Configuration settings for TwistEd - Severe Weather Alerts & Education App
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOAA_API_BASE_URL = "https://api.weather.gov"
NOAA_ALERTS_URL = f"{NOAA_API_BASE_URL}/alerts/active?severity=Severe"

# API Headers
HEADERS = {
    "User-Agent": "TwistEdApp/1.0 (kaarya583@gmail.com)",
    "Accept": "application/geo+json"
}

# App Configuration
APP_TITLE = "🌪️ TwistEd: Real-Time Severe Weather Alerts & Education"
PAGE_TITLE = "🌪️ TwistEd - Tornado & Severe Weather Alerts"
LAYOUT = "wide"

# Auto-refresh settings (in milliseconds)
AUTO_REFRESH_INTERVAL = 5 * 60 * 1000  # 5 minutes

# Map Configuration
DEFAULT_MAP_CENTER = [39.5, -98.35]  # Center of US
DEFAULT_MAP_ZOOM = 4
MAP_TILES = "CartoDB positron"

# Alert Severity Configuration
SEVERITY_COLORS = {
    "Severe": "#d62828",
    "Moderate": "#f77f00", 
    "Minor": "#fcbf49",
    "Extreme": "#9d0208"
}

SEVERITY_WEIGHTS = {
    "Minor": 1, 
    "Moderate": 2, 
    "Severe": 3, 
    "Extreme": 4
}

# Chatbot Configuration
CHATBOT_MODEL = "gpt-4o-mini"
CHATBOT_TEMPERATURE = 0.7
CHATBOT_MAX_TOKENS = 500

# Data Configuration
NOAA_DATA_DIR = "noaa_data"
EMBEDDINGS_FILE = "embeddings.npz"

# Proximity settings
DEFAULT_PROXIMITY_MILES = 50

# Educational content categories
WEATHER_CATEGORIES = [
    "🌪️ Tornadoes",
    "⛈️ Thunderstorms", 
    "🌊 Flash Floods",
    "❄️ Winter Storms",
    "🌀 Hurricanes"
]

# Quick questions for chatbot
QUICK_QUESTIONS = [
    "What causes tornadoes?",
    "How do I stay safe during a thunderstorm?",
    "What's the difference between a tornado watch and warning?",
    "How do hurricanes form?",
    "What should I do during a flash flood?",
    "How do I prepare for severe weather?"
]

# Export settings
EXPORT_FILENAME = "twisted_alerts_export.csv"

# Safety disclaimer
SAFETY_DISCLAIMER = """
⚠️ **Important Safety Notice**: 
This application provides educational information and real-time weather data for informational purposes only. 
Always follow official weather warnings and evacuation orders from local authorities. 
Never rely solely on this app for life-threatening weather decisions.
""" 