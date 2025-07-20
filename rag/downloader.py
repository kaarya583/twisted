import os
import requests
from bs4 import BeautifulSoup

def download_noaa_data(years=[2020], output_dir="noaa_data"):
    base_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Scrape the NOAA page to find available files
    try:
        r = requests.get(base_url)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch directory listing: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    links = [a['href'] for a in soup.find_all('a', href=True)]
    detail_files = [f for f in links if f.startswith("StormEvents_details") and f.endswith(".csv.gz")]

    # Step 2: Download files matching the requested years
    for year in years:
        # Match based on exact filename pattern for year
        matched_files = [f for f in detail_files if f"_{year}.csv.gz" in f]
        if not matched_files:
            print(f"⚠️ No data file found for year {year}")
            continue
        for file_name in matched_files:
            url = base_url + file_name
            dest_path = os.path.join(output_dir, file_name)
            if not os.path.exists(dest_path):
                try:
                    print(f"⬇️ Downloading {file_name}...")
                    file_resp = requests.get(url)
                    file_resp.raise_for_status()
                    with open(dest_path, 'wb') as f:
                        f.write(file_resp.content)
                    print(f"✅ Downloaded {file_name}")
                except Exception as e:
                    print(f"❌ Failed to download {file_name}: {e}")
            else:
                print(f"✔️ File already exists: {file_name}")
