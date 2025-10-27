#!/usr/bin/env python3
"""
Debug script to check feature names
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ml_prediction_service import MLPredictionService
from services.weather_service import WeatherService
from data.locations import get_location_by_name

def debug_features():
    print("Debugging feature names...")
    
    # Initialize services
    weather_service = WeatherService()
    ml_service = MLPredictionService()
    
    # Get location
    location = get_location_by_name('Manila')
    
    # Get weather data
    weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
    
    # Prepare input data
    input_data = ml_service._prepare_prediction_input(weather_data, location)
    print(f"Input data columns: {input_data.columns.tolist()}")
    
    # Check model feature columns
    print(f"Model feature columns: {ml_service.feature_columns}")
    
    # Check if they match
    if list(input_data.columns) == ml_service.feature_columns:
        print("Feature columns match!")
    else:
        print("Feature columns DO NOT match!")
        print(f"Missing in input: {set(ml_service.feature_columns) - set(input_data.columns)}")
        print(f"Extra in input: {set(input_data.columns) - set(ml_service.feature_columns)}")

if __name__ == "__main__":
    debug_features()
