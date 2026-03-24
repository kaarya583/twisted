import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NOAA_API_BASE_URL = "https://api.weather.gov"
NOAA_ALERTS_URL = f"{NOAA_API_BASE_URL}/alerts/active?severity=Severe"
HEADERS = {
    "User-Agent": "TwistEd/2.0 (educational-project)",
    "Accept": "application/geo+json",
}

PAGE_TITLE = "TwistEd - Severe Weather Intelligence"
LAYOUT = "wide"
CHATBOT_MODEL = "gpt-4o-mini"

SAFETY_DISCLAIMER = (
    "Important: This app is educational and analytical. "
    "Always follow official NOAA/NWS alerts and local emergency instructions."
)
