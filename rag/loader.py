import pandas as pd
import glob
import numpy as np
from rag.downloader import download_noaa_data  # Import your downloader function

def load_noaa_data(data_dir="noaa_data", start_year=2015, end_year=2024):
    # Download missing data files first
    download_noaa_data(years=list(range(start_year, end_year + 1)), output_dir=data_dir)

    files = glob.glob(f"{data_dir}/StormEvents_details-*.csv.gz")
    if not files:
        raise FileNotFoundError(f"No NOAA data files found in '{data_dir}'. Check downloader or path.")
    
    dfs = [pd.read_csv(f, compression='gzip', low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess_events(df):
    def make_chunk(row):
        return (
            f"Event Type: {row['EVENT_TYPE']} | "
            f"Date: {row['BEGIN_DATE_TIME']} | "
            f"State: {row['STATE']} | "
            f"County/Zone: {row['CZ_NAME']} | "
            f"Summary: {row['EVENT_NARRATIVE']}"
        )

    df = df.dropna(subset=['EVENT_NARRATIVE'])
    df['text_chunk'] = df.apply(make_chunk, axis=1)
    return df[['text_chunk']]

def search_index(index, question_embed, k=5):
    D, I = index.search(np.array([question_embed]).astype('float32'), k=k)
    return I[0]