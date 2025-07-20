#!/usr/bin/env python3
"""
Setup script for TwistEd - Severe Weather Alerts & Education App
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    print("🔧 Creating .env file...")
    env_content = """# TwistEd Environment Variables
# Add your OpenAI API key here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize NOAA API settings
# NOAA_API_BASE_URL=https://api.weather.gov
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        print("⚠️  Please edit .env file and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["noaa_data", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def check_dependencies():
    """Check if all dependencies are properly installed"""
    required_packages = [
        "streamlit", "requests", "folium", "pandas", 
        "matplotlib", "pgeocode", "openai", "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("\n✅ All dependencies are properly installed")
    return True

def main():
    """Main setup function"""
    print("🌪️ TwistEd Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Setup incomplete. Please install missing packages manually.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run the app: streamlit run twisted.py")
    print("3. Open your browser to the URL shown in the terminal")
    
    print("\n🚀 To start the app, run:")
    print("   streamlit run twisted.py")

if __name__ == "__main__":
    main() 