import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import time
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.database import db_manager

class MLPredictionService:
    """Service for ML-based class suspension prediction"""
    
    def __init__(self, model_path: str = 'windy_rf_model.pkl'):
        self.model = None
        self.feature_columns = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            print(f"ML model loaded successfully (Accuracy: {model_data.get('accuracy', 'Unknown'):.3f})")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            print("Training new model...")
            self.train_model()
    
    def train_model(self):
        """Train a new ML model if loading fails"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
       
        df = self._generate_training_data(3000)
        
      
        df_processed = self._prepare_features(df)
        
        
        X = df_processed.drop('Suspension', axis=1)
        y = df_processed['Suspension'].map({'Yes': 1, 'No': 0})
        
        self.feature_columns = X.columns.tolist()
        
       
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
         
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
       
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now(),
            'accuracy': accuracy
        }
        joblib.dump(model_data, self.model_path)
        
        print(f"New ML model trained and saved (Accuracy: {accuracy:.3f})")
        
        # Log model performance to database
        db_manager.log_model_performance(
            model_type='Random Forest',
            model_version='1.0',
            accuracy=accuracy,
            training_date=datetime.now(),
            feature_count=len(self.feature_columns),
            sample_count=len(df),
            performance_metrics={
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced'
            }
        )
    
    def predict_suspension(self, weather_data: Dict, location_data: Dict, 
                          api_endpoint: str = "", user_agent: str = "", 
                          ip_address: str = "", session_id: str = "") -> Dict:
        """Make ML prediction for class suspension with logging"""
        start_time = time.time()
        error_message = None
        
        try:
            if self.model is None:
                raise Exception("ML model not loaded")
            
            # Prepare input data for prediction
            input_data = self._prepare_prediction_input(weather_data, location_data)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            probability = self.model.predict_proba(input_data)[0]
            
            # Get feature contributions
            feature_contributions = self._get_feature_contributions(input_data)
            
            # Calculate risk level
            # risk_level = self._calculate_risk_level(weather_data, location_data)

            risk_level = 'Low'
            if self._calculate_risk_percentage(weather_data, location_data) > 80:
                risk_level = 'High'
            elif self._calculate_risk_percentage(weather_data, location_data) > 60:
                risk_level = 'Medium'
            elif self._calculate_risk_percentage(weather_data, location_data) > 40:
                risk_level = 'Low'
            
            
        
            
            result = {
                'prediction': 'Suspend Classes' if prediction == 1 else 'Resume Classes',
                'confidence': float(max(probability)),
                'suspend_probability': float(probability[1]),
                'resume_probability': float(probability[0]),
                'risk_level': risk_level,
                'risk_percentage': self._calculate_risk_percentage(weather_data, location_data),
                'feature_contributions': feature_contributions,
                'model_type': 'Random Forest',
                'model_accuracy': getattr(self.model, 'accuracy', 'Unknown'),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Log the prediction to database
            processing_time_ms = int((time.time() - start_time) * 1000)
            db_manager.log_prediction(
                location_name=location_data.get('name', 'Unknown'),
                location_data=location_data,
                weather_data=weather_data,
                prediction_result=result,
                processing_time_ms=processing_time_ms,
                api_endpoint=api_endpoint,
                user_agent=user_agent,
                ip_address=ip_address,
                session_id=session_id,
                error_message=error_message
            )
            
            return result
            
        except Exception as e:
            error_message = str(e)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log the error to database
            db_manager.log_prediction(
                location_name=location_data.get('name', 'Unknown'),
                location_data=location_data,
                weather_data=weather_data,
                prediction_result={},
                processing_time_ms=processing_time_ms,
                api_endpoint=api_endpoint,
                user_agent=user_agent,
                ip_address=ip_address,
                session_id=session_id,
                error_message=error_message
            )
            
            raise e
    
    def _prepare_prediction_input(self, weather_data: Dict, location_data: Dict) -> pd.DataFrame:
        """Prepare input data for ML prediction"""
        current = weather_data.get('current', {})
        

        temp = current.get('temp_c', 0)
        humidity = current.get('humidity', 0)
        rainfall = current.get('precip_mm', 0)
        wind_speed = current.get('wind_kph', 0)
        pressure = current.get('pressure_mb', 1013)
        cloud_cover = current.get('cloud_cover', 0)
        
  
        heat_index = self._calculate_heat_index(temp, humidity)
        comfort_index = self._calculate_comfort_index(temp, humidity, wind_speed)
        
    
        weather_text = current.get('condition', {}).get('text', 'Clear')
        weather_condition = self._convert_weather_condition(weather_text)
        

        location_risk = location_data.get('suspensionRisk', 50)
        risk_factors = location_data.get('riskFactors', [])
        
   
        input_data = {
            'Temperature': temp,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'WindSpeed': wind_speed,
            'Pressure': pressure,
            'CloudCover': cloud_cover,
            'HeatIndex': heat_index,
            'ComfortIndex': comfort_index,
            'WeatherCondition': weather_condition,
            'LocationRisk': location_risk,
            'IsFloodProne': 1 if any('flood' in factor.lower() for factor in risk_factors) else 0,
            'IsHighDensity': 1 if any('density' in factor.lower() for factor in risk_factors) else 0
        }
        
        input_df = pd.DataFrame([input_data])
        return self._prepare_features(input_df)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        df_processed = df.copy()
        

        if 'WeatherCondition' in df_processed.columns:
            weather_dummies = pd.get_dummies(df_processed['WeatherCondition'], prefix='Weather')
            df_processed = pd.concat([df_processed, weather_dummies], axis=1)
            df_processed = df_processed.drop('WeatherCondition', axis=1)
        

        expected_weather_cols = ['Weather_Clear', 'Weather_Clouds', 'Weather_Rain', 
                               'Weather_Thunderstorm', 'Weather_Drizzle', 'Weather_Fog']
        
        for col in expected_weather_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
    
        if self.feature_columns is not None:
            df_processed = df_processed.reindex(columns=self.feature_columns, fill_value=0)
        
        return df_processed
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index"""
        if temp_c < 27:
            return temp_c
        
        temp_f = (temp_c * 9/5) + 32
        
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6
        
        hi_f = (c1 + (c2 * temp_f) + (c3 * humidity) + (c4 * temp_f * humidity) + 
                (c5 * temp_f**2) + (c6 * humidity**2) + (c7 * temp_f**2 * humidity) + 
                (c8 * temp_f * humidity**2) + (c9 * temp_f**2 * humidity**2))
        
        return (hi_f - 32) * 5/9
    
    def _calculate_comfort_index(self, temp_c: float, humidity: float, wind_speed: float) -> float:
        """Calculate comfort index"""
        temp_comfort = max(0, 100 - abs(temp_c - 24) * 5)
        humidity_comfort = max(0, 100 - abs(humidity - 50) * 1.5)
        wind_comfort = max(0, 100 - abs(wind_speed - 5) * 10)
        return (temp_comfort + humidity_comfort + wind_comfort) / 3
    
    def _convert_weather_condition(self, weather_text: str) -> str:
        """Convert weather text to simple condition for ML model"""
        weather_text_lower = weather_text.lower()
        
        if 'rain' in weather_text_lower or 'shower' in weather_text_lower:
            return 'Rain'
        elif 'drizzle' in weather_text_lower:
            return 'Drizzle'
        elif 'thunderstorm' in weather_text_lower or 'storm' in weather_text_lower:
            return 'Thunderstorm'
        elif 'cloud' in weather_text_lower or 'overcast' in weather_text_lower:
            return 'Clouds'
        elif 'fog' in weather_text_lower or 'mist' in weather_text_lower:
            return 'Fog'
        elif 'snow' in weather_text_lower:
            return 'Snow'
        else:
            return 'Clear'
    
    def _calculate_risk_level(self, weather_data: Dict, location_data: Dict) -> str:
        """Calculate risk level based on weather and location"""
        current = weather_data.get('current', {})
        
        temp = current.get('temp_c', 0)
        humidity = current.get('humidity', 0)
        rainfall = current.get('precip_mm', 0)
        wind_speed = current.get('wind_kph', 0)
        
        heat_index = self._calculate_heat_index(temp, humidity)
        location_risk = location_data.get('suspensionRisk', 50)
        
        
        if heat_index > 45 or rainfall > 25 or wind_speed > 65:
            return 'Critical'
        
        
        if (heat_index > 40 or rainfall > 15 or wind_speed > 45 or 
            location_risk > 80):
            return 'High'
        
            
        if (heat_index > 35 or rainfall > 8 or wind_speed > 30 or 
            location_risk > 60):
            return 'Medium'
        
        return 'Low'
    
    def _calculate_risk_percentage(self, weather_data: Dict, location_data: Dict) -> float:
        
        current = weather_data.get('current', {})
        
        temp = current.get('temp_c', 0)
        humidity = current.get('humidity', 0)
        rainfall = current.get('precip_mm', 0)
        wind_speed = current.get('wind_kph', 0)
        
        heat_index = self._calculate_heat_index(temp, humidity)
        location_risk = location_data.get('suspensionRisk', 50)
        
        
        risk_score = location_risk * 0.3
        
        #   
        if heat_index > 45:
            risk_score += 40
        elif heat_index > 40:
            risk_score += 25
        elif heat_index > 35:
            risk_score += 15
        
        if rainfall > 25:
            risk_score += 35
        elif rainfall > 15:
            risk_score += 25
        elif rainfall > 8:
            risk_score += 15
        
        if wind_speed > 65:
            risk_score += 30
        elif wind_speed > 45:
            risk_score += 20
        elif wind_speed > 30:
            risk_score += 10
        
        return min(100, max(0, risk_score))
    
    def _get_feature_contributions(self, input_data: pd.DataFrame) -> Dict:
        
        if self.model is None:
            return {}
        
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
            
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {feature: round(importance * 100, 1) for feature, importance in top_features}
    
    def _generate_dynamic_risk_factors(self, weather_data: Dict, location_data: Dict) -> List[str]:
        """Generate dynamic risk factors based on current weather conditions"""
        current = weather_data.get('current', {})
        
        temp = current.get('temp_c', 0)
        humidity = current.get('humidity', 0)
        rainfall = current.get('precip_mm', 0)
        wind_speed = current.get('wind_kph', 0)
        cloud_cover = current.get('cloud_cover', 0)
        pressure = current.get('pressure_mb', 1013)
        
        heat_index = self._calculate_heat_index(temp, humidity)
        comfort_index = self._calculate_comfort_index(temp, humidity, wind_speed)
        
        risk_factors = []
        
        # Temperature-based factors
        if temp > 35:
            risk_factors.append(f'High temperature ({temp:.1f}°C)')
        elif temp > 30:
            risk_factors.append(f'Elevated temperature ({temp:.1f}°C)')
        
        # Heat index factors
        if heat_index > 45:
            risk_factors.append(f'Extreme heat index ({heat_index:.1f}°C)')
        elif heat_index > 40:
            risk_factors.append(f'High heat index ({heat_index:.1f}°C)')
        elif heat_index > 35:
            risk_factors.append(f'Elevated heat index ({heat_index:.1f}°C)')
        
        # Rainfall factors
        if rainfall > 25:
            risk_factors.append(f'Heavy rainfall ({rainfall:.1f}mm/hr)')
        elif rainfall > 15:
            risk_factors.append(f'Moderate rainfall ({rainfall:.1f}mm/hr)')
        elif rainfall > 8:
            risk_factors.append(f'Light rainfall ({rainfall:.1f}mm/hr)')
        elif rainfall > 0:
            risk_factors.append(f'Precipitation ({rainfall:.1f}mm/hr)')
        
        # Wind factors
        if wind_speed > 65:
            risk_factors.append(f'Strong winds ({wind_speed:.1f} km/h)')
        elif wind_speed > 45:
            risk_factors.append(f'High winds ({wind_speed:.1f} km/h)')
        elif wind_speed > 30:
            risk_factors.append(f'Moderate winds ({wind_speed:.1f} km/h)')
        
        # Humidity factors
        if humidity > 85:
            risk_factors.append(f'Very high humidity ({humidity:.0f}%)')
        elif humidity < 40 and temp > 28:
            risk_factors.append(f'Low humidity ({humidity:.0f}%) - drying conditions')
        
        # Comfort factors
        if comfort_index < 40:
            risk_factors.append(f'Poor comfort conditions')
        
        # Cloud cover factors
        if cloud_cover > 90:
            risk_factors.append(f'Heavy cloud cover ({cloud_cover:.0f}%)')
        
        # Pressure factors
        if pressure < 1000:
            risk_factors.append(f'Low pressure system ({pressure:.0f} mb)')
        
        # Add location-specific factors
        base_risk_factors = location_data.get('riskFactors', [])
        if base_risk_factors:
            # Take first 2-3 base factors if no weather factors found
            if len(risk_factors) < 3:
                risk_factors.extend(base_risk_factors[:3 - len(risk_factors)])
        
        return risk_factors[:5]  # Return top 5 risk factors
    
    def _generate_training_data(self, num_samples: int) -> pd.DataFrame:
        """Generate training data for model training"""
        np.random.seed(42)
        
        
        locations = [
            {'name': 'Manila', 'suspensionRisk': 85, 'riskFactors': ['High population density', 'Frequent flooding']},
            {'name': 'Quezon City', 'suspensionRisk': 78, 'riskFactors': ['Largest city', 'High student population']},
            {'name': 'Malabon', 'suspensionRisk': 88, 'riskFactors': ['Severe flooding', 'Low-lying area']},
            {'name': 'Navotas', 'suspensionRisk': 90, 'riskFactors': ['Severe coastal flooding', 'Storm surge risk']},
            {'name': 'Makati', 'suspensionRisk': 45, 'riskFactors': ['Business district', 'Better infrastructure']},
        ]
        
        data = []
        for _ in range(num_samples):
            location = np.random.choice(locations)
            location_risk = location['suspensionRisk']
            
            
            temp = np.random.normal(28, 4)
            humidity = np.random.normal(70, 15)
            humidity = max(30, min(95, humidity))
            
            rainfall = np.random.exponential(3)
            if location_risk > 70:
                rainfall += np.random.exponential(2)
            
            wind_speed = np.random.gamma(2, 2)
            pressure = np.random.normal(1010, 12)
            cloud_cover = np.random.randint(0, 100)
            
            heat_index = self._calculate_heat_index(temp, humidity)
            comfort_index = self._calculate_comfort_index(temp, humidity, wind_speed)
            
            weather_condition = self._determine_weather_condition(rainfall, humidity, cloud_cover)
            
            
            suspension = self._determine_suspension(
                temp, humidity, rainfall, wind_speed, 
                heat_index, comfort_index, weather_condition, location_risk
            )
            
            data.append({
                'Temperature': temp,
                'Humidity': humidity,
                'Rainfall': rainfall,
                'WindSpeed': wind_speed,
                'Pressure': pressure,
                'CloudCover': cloud_cover,
                'HeatIndex': heat_index,
                'ComfortIndex': comfort_index,
                'WeatherCondition': weather_condition,
                'LocationRisk': location_risk,
                'IsFloodProne': 1 if any('flood' in factor.lower() for factor in location['riskFactors']) else 0,
                'IsHighDensity': 1 if any('density' in factor.lower() for factor in location['riskFactors']) else 0,
                'Suspension': suspension
            })
        
        return pd.DataFrame(data)
    
    def _determine_weather_condition(self, rainfall: float, humidity: float, cloud_cover: float) -> str:
        """Determine weather condition"""
        if rainfall > 15:
            return 'Rain'
        elif rainfall > 8:
            return 'Rain'
        elif rainfall > 3:
            return 'Drizzle'
        elif cloud_cover > 80:
            return 'Clouds'
        elif humidity > 85:
            return 'Fog'
        else:
            return 'Clear'
    
    def _determine_suspension(self, temp: float, humidity: float, rainfall: float, 
                            wind_speed: float, heat_index: float, comfort_index: float, 
                            condition: str, location_risk: float) -> str:
        
        
        if heat_index > 45 or rainfall > 25 or wind_speed > 65:
            return 'Yes'
        
        # Weighted risk factors
        risk_score = 0
        if temp > 35: risk_score += 2
        if heat_index > 40: risk_score += 2
        if rainfall > 15: risk_score += 3
        if rainfall > 8: risk_score += 2
        if wind_speed > 45: risk_score += 2
        if comfort_index < 40: risk_score += 1
        if condition in ['Rain', 'Thunderstorm']: risk_score += 1
        
        
        location_factor = location_risk / 50
        total_score = risk_score * location_factor
        
        return 'Yes' if total_score >= 3 else 'No'
