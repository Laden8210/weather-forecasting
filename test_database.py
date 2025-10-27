#!/usr/bin/env python3
"""
Test script for database functionality
"""

import sys
import os
import json
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import db_manager

def test_database():
    """Test database functionality"""
    print("Testing database functionality...")
    
    # Test logging a prediction
    test_location = {
        'name': 'Test City',
        'district': 'Test District',
        'lat': 14.5995,
        'lon': 120.9842,
        'suspensionRisk': 75
    }
    
    test_weather = {
        'current': {
            'temp_c': 28.5,
            'humidity': 80,
            'precip_mm': 5.2,
            'wind_kph': 15,
            'condition': {'text': 'Light Rain'}
        }
    }
    
    test_prediction = {
        'prediction': 'Suspend Classes',
        'confidence': 0.85,
        'suspend_probability': 0.85,
        'resume_probability': 0.15,
        'risk_level': 'High',
        'risk_percentage': 78.5,
        'feature_contributions': {
            'Rainfall': 25.3,
            'LocationRisk': 20.1,
            'Humidity': 15.7
        },
        'model_type': 'Random Forest',
        'model_accuracy': 0.92,
        'prediction_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Test logging
        log_id = db_manager.log_prediction(
            location_name=test_location['name'],
            location_data=test_location,
            weather_data=test_weather,
            prediction_result=test_prediction,
            processing_time_ms=150,
            api_endpoint='/test',
            user_agent='Test Agent',
            ip_address='127.0.0.1',
            session_id='test-session-123'
        )
        
        print(f"✅ Logged prediction with ID: {log_id}")
        
        # Test retrieving logs
        logs = db_manager.get_prediction_logs(limit=5)
        print(f"✅ Retrieved {len(logs)} logs")
        
        # Test statistics
        stats = db_manager.get_prediction_statistics()
        print(f"✅ Retrieved statistics: {json.dumps(stats, indent=2)}")
        
        print("🎉 All database tests passed!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database()
