#!/usr/bin/env python3
"""
Launcher script for TwistEd - Severe Weather Alerts & Education App
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please run 'python setup.py' first to create the .env file")
        return False
    
    # Check if OPENAI_API_KEY is set
    with open('.env', 'r') as f:
        env_content = f.read()
        if 'your_openai_api_key_here' in env_content:
            print("⚠️  Please update your OpenAI API key in the .env file")
            print("   Edit .env and replace 'your_openai_api_key_here' with your actual API key")
            return False
    
    print("✅ Environment looks good!")
    return True

def run_streamlit():
    """Run the Streamlit app"""
    print("🚀 Starting TwistEd...")
    print("📱 The app will open in your default web browser")
    print("🔄 Auto-refresh is enabled (updates every 5 minutes)")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "twisted.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using TwistEd!")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🌪️ TwistEd Launcher")
    print("=" * 30)
    
    # Check environment
    if not check_environment():
        print("\n💡 Quick setup:")
        print("1. Run: python setup.py")
        print("2. Edit .env file with your OpenAI API key")
        print("3. Run: python run_app.py")
        sys.exit(1)
    
    # Run the app
    run_streamlit()

if __name__ == "__main__":
    main() 