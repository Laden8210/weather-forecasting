#!/usr/bin/env python3
"""
Firebase initialization script for AI prediction logs
Run this script to initialize Firebase and test the connection
"""

import sys
import os
import json
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.firebase_config import firebase_config
from models.database import db_manager

def main():
    """Initialize Firebase and test the connection"""
    print("🔥 Initializing Firebase for AI prediction logs...")
    
    try:
        # Check Firebase initialization
        if not firebase_config.is_initialized():
            print("❌ Firebase not initialized. Please check your credentials.")
            print("\n📝 Setup instructions:")
            print("1. Go to Firebase Console (https://console.firebase.google.com/)")
            print("2. Create a new project or select existing project")
            print("3. Go to Project Settings > Service Accounts")
            print("4. Generate new private key (downloads JSON file)")
            print("5. Set environment variable:")
            print("   export FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/service-account-key.json")
            print("6. Or place the JSON file in the backend directory as 'firebase-service-account.json'")
            return 1
        
        print("✅ Firebase initialized successfully!")
        
        # Test database connection
        if db_manager.test_connection():
            print("✅ Firestore connection test successful!")
        else:
            print("❌ Firestore connection test failed!")
            return 1
        
        # Test logging a sample prediction
        print("\n🧪 Testing prediction logging...")
        
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
        
        if log_id and log_id != "error-id":
            print(f"✅ Successfully logged test prediction: {log_id}")
        else:
            print("❌ Failed to log test prediction")
            return 1
        
        # Test retrieving logs
        print("\n📊 Testing log retrieval...")
        logs = db_manager.get_prediction_logs(limit=5)
        print(f"✅ Retrieved {len(logs)} logs from Firestore")
        
        # Test statistics
        print("\n📈 Testing statistics generation...")
        stats = db_manager.get_prediction_statistics()
        print(f"✅ Generated statistics: {json.dumps(stats, indent=2)}")
        
        print("\n🎉 All Firebase tests passed!")
        print("\n📁 Your Firestore collections:")
        print("   - prediction_logs: Stores AI prediction results")
        print("   - model_performance: Stores model training metrics")
        print("   - feature_importance: Stores feature contribution scores")
        
        return 0
        
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
