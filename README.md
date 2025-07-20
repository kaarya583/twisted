# 🌪️ TwistEd: Real-Time Severe Weather Alerts & Education

A comprehensive educational application that provides real-time severe weather alerts from NOAA and features an intelligent chatbot for weather education.

## 🚀 Features

### 📡 Live Weather Alerts
- Real-time severe weather alerts from NOAA API
- Interactive map with color-coded severity levels
- Filter alerts by state, event type, and proximity to your ZIP code
- Historical alert data retrieval
- Export functionality for data analysis

### 🤖 Weather Expert Chatbot
- AI-powered chatbot with comprehensive weather knowledge
- Real-time integration with current weather conditions
- Educational responses about severe weather phenomena
- Safety tips and emergency preparedness guidance
- Quick question buttons for common weather queries

### 📚 Educational Center
- Comprehensive guides on tornadoes, thunderstorms, flash floods, winter storms, and hurricanes
- Enhanced Fujita Scale explanations
- Warning signs and safety protocols
- Interactive tabs for different weather phenomena

### 🗺️ Interactive Mapping
- Folium-based interactive maps
- Heat map visualization of alert density
- Color-coded severity indicators
- User location marking with ZIP code input

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd twisted
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run twisted.py
   ```

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for chatbot functionality)
- Internet connection (for NOAA API access)

## 🎯 Usage

### Live Alerts Mode
- View real-time severe weather alerts
- Filter by state, event type, or proximity
- Export data for analysis
- Interactive map visualization

### Weather Chatbot Mode
- Ask questions about severe weather
- Get real-time weather context
- Receive safety tips and educational information
- Use quick question buttons for common queries

### Learn Mode
- Comprehensive educational content
- Interactive tabs for different weather phenomena
- Safety guidelines and emergency preparedness
- Visual explanations and examples

### Historical Alerts Mode
- Retrieve historical weather data
- Select custom date ranges
- Analyze past weather patterns

## 🔧 Configuration

### API Keys
- **OpenAI API Key**: Required for chatbot functionality
- **NOAA API**: No key required (public API)

### Customization
- Modify `API_URL` in `twisted.py` for different NOAA endpoints
- Adjust refresh intervals in the auto-refresh configuration
- Customize educational content in the `load_educational_data()` function

## 📊 Data Sources

- **NOAA Weather API**: Real-time and historical weather alerts
- **OpenAI GPT-4**: Chatbot intelligence and responses
- **Educational Content**: Curated meteorological information

## 🏗️ Project Structure

```
twisted/
├── twisted.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables (create this)
├── noaa_data/             # Downloaded NOAA data (auto-created)
├── pages/                 # Additional Streamlit pages
│   └── chatbot.py         # Legacy chatbot implementation
└── rag/                   # RAG (Retrieval-Augmented Generation) components
    ├── chatbot.py         # RAG chatbot implementation
    ├── downloader.py      # NOAA data downloader
    ├── embedder.py        # Text embedding utilities
    ├── loader.py          # Data loading and preprocessing
    └── retriever.py       # FAISS index and retrieval
```

## 🔍 Key Features Explained

### Real-Time Data Integration
The app fetches live weather alerts from NOAA's public API every 5 minutes, providing users with the most current severe weather information.

### Intelligent Chatbot
The chatbot combines:
- Current weather context from NOAA API
- Comprehensive educational knowledge base
- OpenAI's GPT-4 for natural language understanding
- Safety-focused responses

### Interactive Mapping
- **Folium Maps**: Interactive web-based maps
- **Heat Maps**: Visual representation of alert density
- **Color Coding**: Severity-based color schemes
- **User Location**: ZIP code-based proximity filtering

## 🚨 Safety Information

This application provides educational information and real-time weather data. However:
- Always follow official weather warnings and evacuation orders
- Use multiple sources for weather information
- Have an emergency plan and kit ready
- Never rely solely on this app for life-threatening weather decisions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NOAA for providing the weather data API
- OpenAI for the chatbot intelligence
- Streamlit for the web application framework
- The meteorological community for educational content

## 📞 Support

For questions or issues:
- Check the documentation
- Review the code comments
- Open an issue on GitHub

---

**Remember**: This app is for educational purposes. Always follow official weather warnings and emergency instructions during severe weather events. 