"""
Firebase Firestore database models and operations for AI prediction logs
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import sys

# Add config directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.firebase_config import firebase_config

class FirebaseDatabaseManager:
    """Manages Firebase Firestore operations for AI prediction logs"""
    
    def __init__(self):
        self.db = firebase_config.get_firestore_client()
        self.collections = {
            'predictions': 'prediction_logs',
            'performance': 'model_performance',
            'features': 'feature_importance'
        }
    
    def _get_collection(self, collection_name: str):
        """Get Firestore collection reference"""
        if not self.db:
            print("⚠️  Firebase not available, using mock collection")
            return None
        return self.db.collection(self.collections.get(collection_name, collection_name))
    
    def log_prediction(self, 
                      location_name: str,
                      location_data: Dict,
                      weather_data: Dict,
                      prediction_result: Dict,
                      processing_time_ms: int = 0,
                      api_endpoint: str = "",
                      user_agent: str = "",
                      ip_address: str = "",
                      session_id: str = "",
                      error_message: str = None) -> str:
        """Log a prediction result to Firestore"""
        try:
            if not self.db:
                print("⚠️  Firebase not available, skipping log")
                return "mock-id"
            
            # Prepare data for Firestore
            doc_data = {
                'timestamp': datetime.now(),
                'location_name': location_name,
                'location_district': location_data.get('district', ''),
                'location_coordinates': {
                    'lat': location_data.get('lat'),
                    'lon': location_data.get('lon')
                },
                'weather_data': weather_data,
                'prediction_result': prediction_result,
                'model_type': prediction_result.get('model_type', ''),
                'model_accuracy': prediction_result.get('model_accuracy', 0.0),
                'confidence': prediction_result.get('confidence', 0.0),
                'risk_level': prediction_result.get('risk_level', ''),
                'risk_percentage': prediction_result.get('risk_percentage', 0.0),
                'processing_time_ms': processing_time_ms,
                'api_endpoint': api_endpoint,
                'user_agent': user_agent,
                'ip_address': ip_address,
                'session_id': session_id,
                'error_message': error_message,
                'success': error_message is None,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Add document to Firestore
            doc_ref = self._get_collection('predictions').add(doc_data)
            prediction_id = doc_ref[1].id
            
            # Log feature importance if available
            if 'feature_contributions' in prediction_result:
                feature_contributions = prediction_result['feature_contributions']
                for feature_name, importance_score in feature_contributions.items():
                    feature_data = {
                        'prediction_id': prediction_id,
                        'feature_name': feature_name,
                        'importance_score': importance_score,
                        'created_at': datetime.now()
                    }
                    self._get_collection('features').add(feature_data)
            
            print(f"✅ Logged prediction to Firestore: {prediction_id}")
            return prediction_id
            
        except Exception as e:
            print(f"❌ Error logging prediction to Firestore: {e}")
            return "error-id"
    
    def log_model_performance(self, 
                            model_type: str,
                            model_version: str = "1.0",
                            accuracy: float = 0.0,
                            training_date: datetime = None,
                            feature_count: int = 0,
                            sample_count: int = 0,
                            performance_metrics: Dict = None) -> str:
        """Log model performance metrics to Firestore"""
        try:
            if not self.db:
                print("⚠️  Firebase not available, skipping performance log")
                return "mock-id"
            
            doc_data = {
                'timestamp': datetime.now(),
                'model_type': model_type,
                'model_version': model_version,
                'accuracy': accuracy,
                'training_date': training_date or datetime.now(),
                'feature_count': feature_count,
                'sample_count': sample_count,
                'performance_metrics': performance_metrics or {},
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            doc_ref = self._get_collection('performance').add(doc_data)
            performance_id = doc_ref[1].id
            
            print(f"✅ Logged model performance to Firestore: {performance_id}")
            return performance_id
            
        except Exception as e:
            print(f"❌ Error logging model performance to Firestore: {e}")
            return "error-id"
    
    def get_prediction_logs(self, 
                          limit: int = 100,
                          location_name: str = None,
                          start_date: datetime = None,
                          end_date: datetime = None,
                          success_only: bool = True) -> List[Dict]:
        """Retrieve prediction logs with optional filters"""
        try:
            if not self.db:
                print("⚠️  Firebase not available, returning empty logs")
                return []
            
            # Get all documents and filter in Python to avoid index requirements
            docs = self._get_collection('predictions').stream()
            
            # Convert to list of dictionaries and apply filters
            logs = []
            for doc in docs:
                log_data = doc.to_dict()
                log_data['id'] = doc.id
                
                # Apply filters
                if location_name and log_data.get('location_name') != location_name:
                    continue
                
                if start_date:
                    timestamp = log_data.get('timestamp')
                    if isinstance(timestamp, datetime):
                        if timestamp.tzinfo is not None:
                            timestamp = timestamp.replace(tzinfo=None)
                        if timestamp < start_date:
                            continue
                
                if end_date:
                    timestamp = log_data.get('timestamp')
                    if isinstance(timestamp, datetime):
                        if timestamp.tzinfo is not None:
                            timestamp = timestamp.replace(tzinfo=None)
                        if timestamp > end_date:
                            continue
                
                if success_only and not log_data.get('success', False):
                    continue
                
                # Convert Firestore timestamps to ISO strings
                for key, value in log_data.items():
                    if isinstance(value, datetime):
                        log_data[key] = value.isoformat()
                
                logs.append(log_data)
            
            # Sort by timestamp descending and apply limit
            logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            logs = logs[:limit]
            
            print(f"✅ Retrieved {len(logs)} logs from Firestore")
            return logs
            
        except Exception as e:
            print(f"❌ Error retrieving prediction logs from Firestore: {e}")
            return []
    
    def get_prediction_statistics(self) -> Dict:
        """Get statistics about predictions"""
        try:
            if not self.db:
                print("⚠️  Firebase not available, returning empty statistics")
                return {}
            
            # Get all prediction logs
            docs = self._get_collection('predictions').stream()
            
            total_predictions = 0
            successful_predictions = 0
            risk_level_stats = {}
            confidence_scores = []
            location_counts = {}
            recent_predictions = 0
            
            # Calculate cutoff for recent predictions (last 24 hours)
            recent_cutoff = datetime.now()
            if recent_cutoff.tzinfo is None:
                recent_cutoff = recent_cutoff.replace(tzinfo=None)
            
            for doc in docs:
                data = doc.to_dict()
                total_predictions += 1
                
                if data.get('success', False):
                    successful_predictions += 1
                    
                    # Risk level distribution
                    risk_level = data.get('risk_level', 'Unknown')
                    risk_level_stats[risk_level] = risk_level_stats.get(risk_level, 0) + 1
                    
                    # Confidence scores
                    confidence = data.get('confidence', 0)
                    if confidence > 0:
                        confidence_scores.append(confidence)
                    
                    # Location counts
                    location = data.get('location_name', 'Unknown')
                    location_counts[location] = location_counts.get(location, 0) + 1
                
                # Recent predictions (last 24 hours)
                timestamp = data.get('timestamp')
                if isinstance(timestamp, datetime):
                    # Ensure both timestamps are timezone-naive for comparison
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)
                    if timestamp >= recent_cutoff - timedelta(hours=24):
                        recent_predictions += 1
            
            # Calculate statistics
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Get top 10 locations
            top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_locations = [{'location': loc, 'count': count} for loc, count in top_locations]
            
            stats = {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'success_rate': (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0,
                'risk_level_distribution': risk_level_stats,
                'average_confidence': round(avg_confidence, 2),
                'top_locations': top_locations,
                'recent_predictions_24h': recent_predictions,
                'last_updated': datetime.now().isoformat()
            }
            
            print(f"✅ Generated statistics from Firestore: {total_predictions} total predictions")
            return stats
            
        except Exception as e:
            print(f"❌ Error getting prediction statistics from Firestore: {e}")
            return {}
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """Clean up logs older than specified days"""
        try:
            if not self.db:
                print("⚠️  Firebase not available, skipping cleanup")
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Query for old logs
            old_logs = self._get_collection('predictions').where('timestamp', '<', cutoff_date).stream()
            
            deleted_count = 0
            batch = self.db.batch()
            batch_size = 0
            
            for doc in old_logs:
                batch.delete(doc.reference)
                batch_size += 1
                deleted_count += 1
                
                # Firestore batch limit is 500 operations
                if batch_size >= 500:
                    batch.commit()
                    batch = self.db.batch()
                    batch_size = 0
            
            # Commit remaining operations
            if batch_size > 0:
                batch.commit()
            
            print(f"✅ Cleaned up {deleted_count} old logs from Firestore")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Error cleaning up old logs from Firestore: {e}")
            return 0
    
    def test_connection(self) -> bool:
        """Test Firebase connection"""
        try:
            if not self.db:
                return False
            
            # Try to read from a collection
            test_docs = self._get_collection('predictions').limit(1).stream()
            list(test_docs)  # Force evaluation
            return True
            
        except Exception as e:
            print(f"❌ Firebase connection test failed: {e}")
            return False

# Global database manager instance
db_manager = FirebaseDatabaseManager()
