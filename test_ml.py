#!/usr/bin/env python3
"""
Test script to debug ML model feature issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ml_prediction_service import MLPredictionService
from services.weather_service import WeatherService
from data.locations import get_location_by_name

def test_ml_prediction():
    print("Testing ML Prediction Service...")
    
    # Initialize services
    weather_service = WeatherService()
    ml_service = MLPredictionService()
    
    # Get location
    location = get_location_by_name('Manila')
    print(f"Location: {location}")
    
    # Get weather data
    weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
    print(f"Weather data keys: {weather_data.keys()}")
    print(f"Current weather: {weather_data['current']}")
    
    # Test prediction
    try:
        prediction = ml_service.predict_suspension(weather_data, location)
        print(f"Prediction successful: {prediction}")
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_prediction()
