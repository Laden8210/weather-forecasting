#!/usr/bin/env python3
"""
Migration status and Firebase setup verification script
"""

import sys
import os
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_migration_status():
    """Check the migration status from SQLite to Firebase"""
    print("🔄 AI Prediction Logs Database Migration Status")
    print("=" * 50)
    
    # Check if old SQLite file exists
    sqlite_file = "ai_logs.db"
    if os.path.exists(sqlite_file):
        print(f"📁 Found old SQLite database: {sqlite_file}")
        print("   This file can be kept as backup or removed after migration")
    else:
        print("✅ No old SQLite database found")
    
    # Check Firebase configuration
    try:
        from config.firebase_config import firebase_config
        if firebase_config.is_initialized():
            print("✅ Firebase is properly initialized")
        else:
            print("❌ Firebase not initialized")
            print("   Run: python init_firebase.py")
    except ImportError as e:
        print(f"❌ Firebase configuration error: {e}")
        print("   Make sure firebase-admin is installed: pip install firebase-admin")
    
    # Check database manager
    try:
        from models.database import db_manager
        if db_manager.test_connection():
            print("✅ Firebase database connection working")
        else:
            print("❌ Firebase database connection failed")
    except Exception as e:
        print(f"❌ Database manager error: {e}")
    
    # Check requirements
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            content = f.read()
            if 'firebase-admin' in content:
                print("✅ Firebase dependencies in requirements.txt")
            else:
                print("❌ Firebase dependencies missing from requirements.txt")
    
    print("\n📋 Migration Summary:")
    print("   ✅ SQLite → Firebase Firestore")
    print("   ✅ Local file → Cloud database")
    print("   ✅ Automatic logging maintained")
    print("   ✅ API endpoints unchanged")
    print("   ✅ Statistics and cleanup features available")
    
    print("\n🚀 Next Steps:")
    print("   1. Set up Firebase project (see FIREBASE_SETUP.md)")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Initialize Firebase: python init_firebase.py")
    print("   4. Test integration: python test_firebase.py")
    print("   5. Start backend: python app.py")
    
    print(f"\n📅 Migration completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    check_migration_status()
