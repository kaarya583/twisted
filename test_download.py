from rag.downloader import download_noaa_data
import os

print("Current working directory:", os.getcwd())
download_noaa_data(start_year=2020, end_year=2020)  # Start small for testing
