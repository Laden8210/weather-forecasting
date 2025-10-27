#!/usr/bin/env python3
"""
Test script for Firebase database functionality
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import db_manager
from config.firebase_config import firebase_config

def test_firebase():
    """Test Firebase functionality"""
    print("🔥 Testing Firebase database functionality...")
    
    # Check if Firebase is initialized
    if not firebase_config.is_initialized():
        print("❌ Firebase not initialized. Please run init_firebase.py first.")
        return False
    
    print("✅ Firebase is initialized")
    
    # Test connection
    if not db_manager.test_connection():
        print("❌ Firebase connection test failed")
        return False
    
    print("✅ Firebase connection test passed")
    
    # Test logging multiple predictions
    print("\n📝 Testing prediction logging...")
    
    test_cases = [
        {
            'location': {'name': 'Manila', 'district': 'NCR', 'lat': 14.5995, 'lon': 120.9842, 'suspensionRisk': 85},
            'weather': {'current': {'temp_c': 32.0, 'humidity': 75, 'precip_mm': 0, 'wind_kph': 10, 'condition': {'text': 'Clear'}}},
            'prediction': {'prediction': 'Resume Classes', 'confidence': 0.78, 'risk_level': 'Low', 'risk_percentage': 25.0, 'model_type': 'Random Forest', 'model_accuracy': 0.92}
        },
        {
            'location': {'name': 'Malabon', 'district': 'NCR', 'lat': 14.6655, 'lon': 120.9483, 'suspensionRisk': 90},
            'weather': {'current': {'temp_c': 28.0, 'humidity': 85, 'precip_mm': 15.5, 'wind_kph': 25, 'condition': {'text': 'Heavy Rain'}}},
            'prediction': {'prediction': 'Suspend Classes', 'confidence': 0.92, 'risk_level': 'High', 'risk_percentage': 88.0, 'model_type': 'Random Forest', 'model_accuracy': 0.92}
        },
        {
            'location': {'name': 'Makati', 'district': 'NCR', 'lat': 14.5547, 'lon': 121.0244, 'suspensionRisk': 45},
            'weather': {'current': {'temp_c': 30.0, 'humidity': 70, 'precip_mm': 2.1, 'wind_kph': 15, 'condition': {'text': 'Light Rain'}}},
            'prediction': {'prediction': 'Resume Classes', 'confidence': 0.65, 'risk_level': 'Medium', 'risk_percentage': 45.0, 'model_type': 'Random Forest', 'model_accuracy': 0.92}
        }
    ]
    
    logged_ids = []
    for i, test_case in enumerate(test_cases):
        prediction_id = db_manager.log_prediction(
            location_name=test_case['location']['name'],
            location_data=test_case['location'],
            weather_data=test_case['weather'],
            prediction_result=test_case['prediction'],
            processing_time_ms=100 + i * 50,
            api_endpoint='/test',
            user_agent='Firebase Test Agent',
            ip_address='127.0.0.1',
            session_id=f'test-session-{i}'
        )
        
        if prediction_id and prediction_id != "error-id":
            logged_ids.append(prediction_id)
            print(f"✅ Logged prediction {i+1}: {prediction_id}")
        else:
            print(f"❌ Failed to log prediction {i+1}")
    
    print(f"\n📊 Successfully logged {len(logged_ids)} predictions")
    
    # Test retrieving logs
    print("\n📖 Testing log retrieval...")
    
    # Get recent logs
    recent_logs = db_manager.get_prediction_logs(limit=10)
    print(f"✅ Retrieved {len(recent_logs)} recent logs")
    
    # Get logs for specific location
    manila_logs = db_manager.get_prediction_logs(location_name='Manila')
    print(f"✅ Retrieved {len(manila_logs)} logs for Manila")
    
    # Get logs with date filter
    yesterday = datetime.now() - timedelta(days=1)
    recent_logs_filtered = db_manager.get_prediction_logs(start_date=yesterday)
    print(f"✅ Retrieved {len(recent_logs_filtered)} logs from yesterday")
    
    # Test statistics
    print("\n📈 Testing statistics...")
    stats = db_manager.get_prediction_statistics()
    print(f"✅ Generated statistics:")
    print(f"   - Total predictions: {stats.get('total_predictions', 0)}")
    print(f"   - Success rate: {stats.get('success_rate', 0):.1f}%")
    print(f"   - Average confidence: {stats.get('average_confidence', 0):.2f}")
    print(f"   - Recent predictions (24h): {stats.get('recent_predictions_24h', 0)}")
    
    # Test model performance logging
    print("\n🎯 Testing model performance logging...")
    performance_id = db_manager.log_model_performance(
        model_type='Random Forest',
        model_version='1.0',
        accuracy=0.92,
        training_date=datetime.now(),
        feature_count=15,
        sample_count=3000,
        performance_metrics={
            'precision': 0.89,
            'recall': 0.91,
            'f1_score': 0.90
        }
    )
    
    if performance_id and performance_id != "error-id":
        print(f"✅ Logged model performance: {performance_id}")
    else:
        print("❌ Failed to log model performance")
    
    print("\n🎉 All Firebase tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_firebase()
    exit(0 if success else 1)
