"""
Feature Engineering Module for TwistEd ML Pipeline
Creates ML-ready features from NOAA weather data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon, Point
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering pipeline for weather data
    Creates geographic, temporal, and text-based features
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.pca = None
        self.feature_names = []
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Raw NOAA weather data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Starting feature engineering...")
        
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # 1. Temporal features
        df_features = self._create_temporal_features(df_features)
        
        # 2. Geographic features  
        df_features = self._create_geographic_features(df_features)
        
        # 3. Text features
        df_features = self._create_text_features(df_features)
        
        # 4. Event type features
        df_features = self._create_event_features(df_features)
        
        # 5. Severity features
        df_features = self._create_severity_features(df_features)
        
        # 6. Interaction features
        df_features = self._create_interaction_features(df_features)
        
        print(f"✅ Feature engineering complete! Created {len(df_features.columns)} features")
        return df_features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        print("  📅 Creating temporal features...")
        
        # Convert to datetime
        df['BEGIN_DATE_TIME'] = pd.to_datetime(df['BEGIN_DATE_TIME'], errors='coerce')
        df['END_DATE_TIME'] = pd.to_datetime(df['END_DATE_TIME'], errors='coerce')
        
        # Extract time components
        df['year'] = df['BEGIN_DATE_TIME'].dt.year
        df['month'] = df['BEGIN_DATE_TIME'].dt.month
        df['day_of_year'] = df['BEGIN_DATE_TIME'].dt.dayofyear
        df['day_of_week'] = df['BEGIN_DATE_TIME'].dt.dayofweek
        df['hour'] = df['BEGIN_DATE_TIME'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Seasonal features
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Event duration
        df['event_duration_hours'] = (
            df['END_DATE_TIME'] - df['BEGIN_DATE_TIME']
        ).dt.total_seconds() / 3600
        
        # Fill missing duration with median
        df['event_duration_hours'] = df['event_duration_hours'].fillna(
            df['event_duration_hours'].median()
        )
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        print("  🌍 Creating geographic features...")
        
        # State encoding
        df['state_encoded'] = pd.Categorical(df['STATE']).codes
        
        # County/zone encoding
        df['county_encoded'] = pd.Categorical(df['CZ_NAME']).codes
        
        # Region features (based on state groupings)
        northeast_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA']
        southeast_states = ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA']
        midwest_states = ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS']
        southwest_states = ['OK', 'TX', 'NM', 'AZ']
        west_states = ['CO', 'WY', 'MT', 'ID', 'WA', 'OR', 'CA', 'NV', 'UT']
        
        df['region_northeast'] = df['STATE'].isin(northeast_states).astype(int)
        df['region_southeast'] = df['STATE'].isin(southeast_states).astype(int)
        df['region_midwest'] = df['STATE'].isin(midwest_states).astype(int)
        df['region_southwest'] = df['STATE'].isin(southwest_states).astype(int)
        df['region_west'] = df['STATE'].isin(west_states).astype(int)
        
        # Latitude/longitude features (if available)
        if 'BEGIN_LAT' in df.columns and 'BEGIN_LON' in df.columns:
            df['latitude'] = pd.to_numeric(df['BEGIN_LAT'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['BEGIN_LON'], errors='coerce')
            
            # Fill missing coordinates with state centroids (approximate)
            state_centroids = {
                'TX': (31.9686, -99.9018), 'CA': (36.7783, -119.4179),
                'FL': (27.6648, -81.5158), 'NY': (42.1657, -74.9481),
                'PA': (40.5908, -77.2098), 'IL': (40.6331, -89.3985),
                'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001),
                'NC': (35.7596, -79.0193), 'MI': (44.3148, -85.6024)
            }
            
            for state, (lat, lon) in state_centroids.items():
                mask = (df['STATE'] == state) & (df['latitude'].isna())
                df.loc[mask, 'latitude'] = lat
                df.loc[mask, 'longitude'] = lon
            
            # Geographic clustering features
            df['lat_bin'] = pd.cut(df['latitude'], bins=10, labels=False)
            df['lon_bin'] = pd.cut(df['longitude'], bins=10, labels=False)
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features from event narratives"""
        print("  📝 Creating text features...")
        
        # Clean text data
        df['clean_narrative'] = df['EVENT_NARRATIVE'].fillna('').astype(str)
        df['clean_narrative'] = df['clean_narrative'].str.lower()
        df['clean_narrative'] = df['clean_narrative'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Text length features
        df['narrative_length'] = df['clean_narrative'].str.len()
        df['word_count'] = df['clean_narrative'].str.split().str.len()
        
        # TF-IDF features for key weather terms
        weather_keywords = [
            'tornado', 'thunderstorm', 'flood', 'hurricane', 'snow', 'ice',
            'wind', 'hail', 'lightning', 'rain', 'storm', 'warning', 'watch'
        ]
        
        for keyword in weather_keywords:
            df[f'contains_{keyword}'] = df['clean_narrative'].str.contains(keyword).astype(int)
        
        # TF-IDF vectorization (if enough data)
        if len(df) > 100:
            try:
                # Use only non-empty narratives
                non_empty_mask = df['clean_narrative'].str.len() > 10
                if non_empty_mask.sum() > 50:
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=50,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                    
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                        df.loc[non_empty_mask, 'clean_narrative']
                    )
                    
                    # Reduce dimensionality with PCA
                    self.pca = PCA(n_components=10)
                    tfidf_pca = self.pca.fit_transform(tfidf_matrix.toarray())
                    
                    # Create feature names
                    for i in range(10):
                        df[f'tfidf_pca_{i}'] = 0
                    
                    # Fill in the values
                    df.loc[non_empty_mask, [f'tfidf_pca_{i}' for i in range(10)]] = tfidf_pca
                    
                    print(f"    ✅ Created {tfidf_pca.shape[1]} TF-IDF PCA features")
            except Exception as e:
                print(f"    ⚠️ TF-IDF feature creation failed: {e}")
        
        return df
    
    def _create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create event type features"""
        print("  🎯 Creating event features...")
        
        # Event type encoding
        df['event_type_encoded'] = pd.Categorical(df['EVENT_TYPE']).codes
        
        # Create dummy variables for major event types
        major_events = ['Tornado', 'Thunderstorm Wind', 'Hail', 'Flash Flood', 'Flood']
        for event in major_events:
            df[f'event_{event.lower().replace(" ", "_")}'] = (
                df['EVENT_TYPE'] == event
            ).astype(int)
        
        # Event category grouping
        wind_events = ['Thunderstorm Wind', 'High Wind', 'Strong Wind']
        water_events = ['Flash Flood', 'Flood', 'Heavy Rain']
        ice_events = ['Winter Storm', 'Ice Storm', 'Sleet']
        
        df['event_category_wind'] = df['EVENT_TYPE'].isin(wind_events).astype(int)
        df['event_category_water'] = df['EVENT_TYPE'].isin(water_events).astype(int)
        df['event_category_ice'] = df['EVENT_TYPE'].isin(ice_events).astype(int)
        
        return df
    
    def _create_severity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create severity-related features"""
        print("  ⚠️ Creating severity features...")
        
        # Severity encoding
        severity_mapping = {
            'Minor': 1, 'Moderate': 2, 'Severe': 3, 'Extreme': 4
        }
        df['severity_encoded'] = df['severity'].map(severity_mapping).fillna(1)
        
        # Urgency encoding - handle both string and numeric values
        if 'urgency' in df.columns:
            if df['urgency'].dtype == 'object':
                urgency_mapping = {
                    'Unknown': 0, 'Expected': 1, 'Immediate': 2
                }
                df['urgency_encoded'] = df['urgency'].map(urgency_mapping).fillna(0)
            else:
                df['urgency_encoded'] = df['urgency'].fillna(0)
        else:
            df['urgency_encoded'] = 0
        
        # Create severity dummies
        for severity in ['Minor', 'Moderate', 'Severe', 'Extreme']:
            df[f'severity_{severity.lower()}'] = (df['severity'] == severity).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different feature types"""
        print("  🔗 Creating interaction features...")
        
        # Time-Event interactions
        df['spring_tornado'] = df['is_spring'] * df['event_tornado']
        df['summer_thunderstorm'] = df['is_summer'] * df['event_thunderstorm_wind']
        df['winter_ice'] = df['is_winter'] * df['event_category_ice']
        
        # Geographic-Event interactions
        if 'region_southeast' in df.columns:
            df['southeast_tornado'] = df['region_southeast'] * df['event_tornado']
            df['midwest_thunderstorm'] = df['region_midwest'] * df['event_thunderstorm_wind']
        
        # Severity-Time interactions
        df['severe_weekend'] = df['is_weekend'] * (df['severity_encoded'] >= 3).astype(int)
        df['extreme_business_hours'] = (~df['is_weekend']) * (df['severity_encoded'] == 4).astype(int)
        
        return df
    
    def get_feature_names(self) -> list:
        """Get list of engineered feature names"""
        return [col for col in self.feature_names if col not in [
            'EVENT_TYPE', 'STATE', 'CZ_NAME', 'EVENT_NARRATIVE', 'BEGIN_DATE_TIME', 'END_DATE_TIME'
        ]]
    
    def save_feature_engineering_info(self, filepath: str):
        """Save feature engineering configuration for reproducibility"""
        import json
        
        config = {
            'tfidf_features': self.tfidf_vectorizer is not None,
            'pca_components': self.pca.n_components_ if self.pca else None,
            'feature_count': len(self.feature_names),
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Feature engineering config saved to {filepath}")
