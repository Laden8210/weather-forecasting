"""
Firebase configuration and authentication setup
"""

import os
import json
from typing import Optional
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client

class FirebaseConfig:
    """Firebase configuration and initialization"""
    
    def __init__(self):
        self.db: Optional[Client] = None
        self.initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check for service account key file in multiple locations
            service_account_path = None
            
            # 1. Check environment variable
            env_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
            if env_path and os.path.exists(env_path):
                service_account_path = env_path
            
            # 2. Check for firebase-service-account.json in current directory
            elif os.path.exists('firebase-service-account.json'):
                service_account_path = 'firebase-service-account.json'
            
            # 3. Check for firebase-service-account.json in backend directory
            elif os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'firebase-service-account.json')):
                service_account_path = os.path.join(os.path.dirname(__file__), '..', 'firebase-service-account.json')
            
            if service_account_path:
                # Use service account file
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                print(f"✅ Firebase initialized with service account: {service_account_path}")
            else:
                # Try to use default credentials (for Google Cloud environments)
                try:
                    firebase_admin.initialize_app()
                    print("✅ Firebase initialized with default credentials")
                except Exception as e:
                    print(f"⚠️  Firebase initialization failed: {e}")
                    print("📝 Please set up Firebase credentials:")
                    print("   1. Download service account key from Firebase Console")
                    print("   2. Place it as 'firebase-service-account.json' in backend directory")
                    print("   3. Or set FIREBASE_SERVICE_ACCOUNT_PATH environment variable")
                    print("   4. Or run in Google Cloud environment with default credentials")
                    return
            
            # Initialize Firestore client
            self.db = firestore.client()
            self.initialized = True
            print("✅ Firestore client initialized")
            
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
            self.initialized = False
    
    def get_firestore_client(self) -> Optional[Client]:
        """Get Firestore client"""
        if not self.initialized:
            print("⚠️  Firebase not initialized. Using mock client.")
            return None
        return self.db
    
    def is_initialized(self) -> bool:
        """Check if Firebase is properly initialized"""
        return self.initialized

# Global Firebase configuration instance
firebase_config = FirebaseConfig()
