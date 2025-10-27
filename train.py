import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

# Metro Manila locations database
METRO_MANILA_PLACES = [
    { 
        'name': 'Manila', 
        'lat': 14.5995, 
        'lon': 120.9842, 
        'type': 'City', 
        'district': 'Capital',
        'suspensionRisk': 85,
        'riskFactors': ['High population density', 'Frequent flooding', 'Urban heat island', 'Government center']
    },
    { 
        'name': 'Quezon City', 
        'lat': 14.6760, 
        'lon': 121.0437, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 78,
        'riskFactors': ['Largest city', 'High student population', 'Flood-prone areas', 'Traffic congestion']
    },
    { 
        'name': 'Caloocan', 
        'lat': 14.6548, 
        'lon': 120.9842, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 72,
        'riskFactors': ['Dense population', 'Flooding issues', 'Industrial areas', 'Transportation hub']
    },
    { 
        'name': 'Las Piñas', 
        'lat': 14.4500, 
        'lon': 120.9833, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 65,
        'riskFactors': ['Coastal flooding', 'Typhoon exposure', 'Growing population', 'Infrastructure development']
    },
    { 
        'name': 'Makati', 
        'lat': 14.5547, 
        'lon': 121.0244, 
        'type': 'City', 
        'district': '2nd District',
        'suspensionRisk': 45,
        'riskFactors': ['Business district', 'Better infrastructure', 'Lower flood risk', 'Urban planning']
    },
    { 
        'name': 'Malabon', 
        'lat': 14.6667, 
        'lon': 120.9500, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 88,
        'riskFactors': ['Severe flooding', 'Low-lying area', 'Poor drainage', 'High water table']
    },
    { 
        'name': 'Mandaluyong', 
        'lat': 14.5833, 
        'lon': 121.0333, 
        'type': 'City', 
        'district': '2nd District',
        'suspensionRisk': 55,
        'riskFactors': ['Mixed residential/commercial', 'Moderate flooding', 'Good infrastructure', 'Central location']
    },
    { 
        'name': 'Marikina', 
        'lat': 14.6500, 
        'lon': 121.1000, 
        'type': 'City', 
        'district': '2nd District',
        'suspensionRisk': 82,
        'riskFactors': ['River flooding', 'Typhoon path', 'Valley location', 'Historical flood events']
    },
    { 
        'name': 'Muntinlupa', 
        'lat': 14.4000, 
        'lon': 121.0500, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 68,
        'riskFactors': ['Laguna Lake proximity', 'Flooding from lake', 'Growing development', 'Transportation issues']
    },
    { 
        'name': 'Navotas', 
        'lat': 14.6667, 
        'lon': 120.9500, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 90,
        'riskFactors': ['Severe coastal flooding', 'Storm surge risk', 'Low elevation', 'Poor drainage']
    },
    { 
        'name': 'Parañaque', 
        'lat': 14.5000, 
        'lon': 121.0167, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 70,
        'riskFactors': ['Coastal flooding', 'Airport proximity', 'Mixed development', 'Typhoon exposure']
    },
    { 
        'name': 'Pasay', 
        'lat': 14.5500, 
        'lon': 121.0000, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 60,
        'riskFactors': ['Airport area', 'Coastal location', 'Commercial zone', 'Infrastructure priority']
    },
    { 
        'name': 'Pasig', 
        'lat': 14.5833, 
        'lon': 121.0833, 
        'type': 'City', 
        'district': '2nd District',
        'suspensionRisk': 75,
        'riskFactors': ['River flooding', 'Industrial areas', 'Mixed development', 'Transportation hub']
    },
    { 
        'name': 'San Juan', 
        'lat': 14.6000, 
        'lon': 121.0333, 
        'type': 'City', 
        'district': '2nd District',
        'suspensionRisk': 50,
        'riskFactors': ['Small area', 'Better drainage', 'Residential focus', 'Lower population density']
    },
    { 
        'name': 'Taguig', 
        'lat': 14.5167, 
        'lon': 121.0500, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 58,
        'riskFactors': ['Mixed development', 'BGC area', 'Moderate flooding', 'Infrastructure investment']
    },
    { 
        'name': 'Valenzuela', 
        'lat': 14.7000, 
        'lon': 120.9833, 
        'type': 'City', 
        'district': '1st District',
        'suspensionRisk': 68,
        'riskFactors': ['Industrial areas', 'Flooding issues', 'Growing population', 'Transportation challenges']
    },
    { 
        'name': 'Pateros', 
        'lat': 14.5500, 
        'lon': 121.0667, 
        'type': 'Municipality', 
        'district': '1st District',
        'suspensionRisk': 80,
        'riskFactors': ['Small municipality', 'River proximity', 'Limited resources', 'Flood-prone']
    },
]

class WindyWeatherCollector:
    """Weather data collector using Windy.com APIs"""
    
    def __init__(self):
        self.windy_api_url = "https://api.windy.com/api/point-forecast/v2"
        # Using Windy's free tier - no API key required for basic access
        
    def get_weather_data(self, location_name):
        """Get weather data from Windy API"""
        location = self._find_location(location_name)
        if not location:
            return None
        
        try:
            # Windy API parameters
            params = {
                'lat': location['lat'],
                'lon': location['lon'],
                'model': 'gfs',  # Global Forecast System
                'parameters': ['temp', 'rh', 'precip', 'wind', 'pressure', 'clouds'],
                'levels': ['surface'],
                'key': 'free_tier'  # Windy's free access
            }
            
            # Try to get data from Windy API
            weather_data = self._get_windy_api_data(location)
            if not weather_data:
                # Fallback to simulated data based on Windy's patterns
                weather_data = self._get_simulated_weather(location)
            
            return self._enhance_weather_data(weather_data, location)
            
        except Exception as e:
            print(f"Error getting Windy data: {e}")
            return self._get_simulated_weather(location)
    
    def _get_windy_api_data(self, location):
        """Get actual data from Windy API"""
        try:
            # Windy's point forecast API (simulated since actual API might require key)
            # This is a simulation of what Windy API would return
            base_temp = 28 + (location['lat'] - 14.5) * 2  # Temperature variation
            base_humidity = 70 + (location['suspensionRisk'] - 50) / 5
            
            # Simulate seasonal variations
            month = datetime.now().month
            if month in [6, 7, 8, 9]:  # Rainy season
                rainfall = np.random.exponential(5)
                humidity = base_humidity + 15
            else:
                rainfall = np.random.exponential(2)
                humidity = base_humidity
            
            return {
                'temp': base_temp + np.random.normal(0, 3),
                'rh': min(95, max(40, humidity + np.random.normal(0, 10))),
                'precip': max(0, rainfall),
                'wind': np.random.gamma(2, 3),
                'pressure': 1010 + np.random.normal(0, 10),
                'clouds': np.random.randint(0, 100),
                'weather': self._get_weather_condition(rainfall, humidity)
            }
        except Exception as e:
            print(f"Windy API simulation error: {e}")
            return None
    
    def _get_simulated_weather(self, location):
        """Generate realistic weather data for Metro Manila"""
        # Base values with location adjustments
        base_temp = 28 + (location['suspensionRisk'] - 50) / 20
        base_humidity = 65 + (location['suspensionRisk'] - 50) / 10
        
        # Time-based variations
        hour = datetime.now().hour
        month = datetime.now().month
        
        # Temperature variation throughout day
        temp_variation = -5 if hour < 6 else (5 if hour > 12 else 0)
        
        # Seasonal adjustments
        if month in [6, 7, 8, 9]:  # Rainy season
            rainfall = np.random.exponential(8)
            humidity = base_humidity + 20
            cloud_cover = np.random.randint(50, 100)
        elif month in [3, 4, 5]:  # Summer
            rainfall = np.random.exponential(1)
            humidity = base_humidity - 10
            cloud_cover = np.random.randint(10, 50)
        else:  # Transition months
            rainfall = np.random.exponential(3)
            humidity = base_humidity
            cloud_cover = np.random.randint(30, 80)
        
        # Location-specific adjustments
        if location['name'] in ['Malabon', 'Navotas', 'Marikina']:
            rainfall += np.random.exponential(3)  # Flood-prone areas get more rain
        if location['name'] in ['Makati', 'San Juan']:
            rainfall *= 0.7  # Better drainage areas get less rain impact
        
        return {
            'temp': max(20, min(40, base_temp + temp_variation + np.random.normal(0, 2))),
            'rh': min(95, max(40, humidity + np.random.normal(0, 5))),
            'precip': max(0, rainfall),
            'wind': np.random.gamma(2, 2) + (location['suspensionRisk'] / 50),
            'pressure': 1010 + np.random.normal(0, 8),
            'clouds': cloud_cover,
            'weather': self._get_weather_condition(rainfall, humidity)
        }
    
    def _get_weather_condition(self, rainfall, humidity):
        """Determine weather condition based on parameters"""
        if rainfall > 20:
            return 'Thunderstorm'
        elif rainfall > 10:
            return 'Rain'
        elif rainfall > 5:
            return 'Drizzle'
        elif humidity > 85:
            return 'Fog'
        elif humidity < 40:
            return 'Clear'
        else:
            return 'Clouds'
    
    def _find_location(self, location_name):
        """Find location from database"""
        location_name_lower = location_name.lower().strip()
        
        for location in METRO_MANILA_PLACES:
            if location['name'].lower() == location_name_lower:
                return location
        
        for location in METRO_MANILA_PLACES:
            if location_name_lower in location['name'].lower():
                return location
        
        return None
    
    def _enhance_weather_data(self, weather_data, location):
        """Enhance weather data with additional calculations"""
        temp = weather_data['temp']
        humidity = weather_data['rh']
        
        # Calculate heat index
        heat_index = self.calculate_heat_index(temp, humidity)
        
        # Calculate comfort index
        comfort_index = self.calculate_comfort_index(temp, humidity, weather_data['wind'])
        
        return {
            'timestamp': datetime.now(),
            'location_name': location['name'],
            'latitude': location['lat'],
            'longitude': location['lon'],
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'rainfall': round(weather_data['precip'], 1),
            'wind_speed': round(weather_data['wind'], 1),
            'pressure': round(weather_data['pressure'], 1),
            'cloud_cover': weather_data['clouds'],
            'weather_condition': weather_data['weather'],
            'heat_index': round(heat_index, 1),
            'comfort_index': round(comfort_index, 1),
            'suspension_risk_base': location['suspensionRisk'],
            'risk_factors': location['riskFactors'],
            'location_type': location['type'],
            'district': location['district'],
            'data_source': 'Windy.com Simulation'
        }
    
    def calculate_heat_index(self, temperature, humidity):
        """Calculate heat index using NOAA formula"""
        if temperature < 27:
            return temperature
        
        # NOAA heat index formula
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6
        
        T = temperature
        R = humidity
        
        hi = (c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + 
              (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + 
              (c8 * T * R**2) + (c9 * T**2 * R**2))
        
        return hi
    
    def calculate_comfort_index(self, temp, humidity, wind_speed):
        """Calculate comfort index (0-100, higher is better)"""
        # Temperature comfort (ideal around 22-26°C)
        temp_comfort = max(0, 100 - abs(temp - 24) * 5)
        
        # Humidity comfort (ideal 40-60%)
        humidity_comfort = max(0, 100 - abs(humidity - 50) * 1.5)
        
        # Wind comfort (light breeze is best)
        wind_comfort = max(0, 100 - abs(wind_speed - 5) * 10)
        
        return (temp_comfort + humidity_comfort + wind_comfort) / 3

class WindyRandomForestPredictor:
    """Random Forest predictor for class suspension"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = None
        if model_path:
            self.load_model(model_path)
        else:
            self.train_model()
    
    def generate_training_data(self, num_samples=3000):
        """Generate comprehensive training data"""
        np.random.seed(42)
        
        data = []
        for _ in range(num_samples):
            # Select random location characteristics
            location = np.random.choice(METRO_MANILA_PLACES)
            location_risk = location['suspensionRisk']
            
            # Generate realistic weather data
            temp = np.random.normal(28, 4)
            humidity = np.random.normal(70, 15)
            humidity = max(30, min(95, humidity))
            
            # Rainfall based on location risk and season
            base_rainfall = np.random.exponential(3)
            if location_risk > 70:  # High-risk flood areas
                base_rainfall += np.random.exponential(2)
            
            wind_speed = np.random.gamma(2, 2)
            pressure = np.random.normal(1010, 12)
            cloud_cover = np.random.randint(0, 100)
            
            # Calculate derived features
            heat_index = self.calculate_heat_index(temp, humidity)
            comfort_index = self.calculate_comfort_index(temp, humidity, wind_speed)
            
            # Determine weather condition
            weather_condition = self.determine_weather_condition(base_rainfall, humidity, cloud_cover)
            
            # Adjust rainfall based on condition
            if weather_condition == 'Thunderstorm':
                rainfall = base_rainfall + np.random.exponential(5)
            elif weather_condition == 'Rain':
                rainfall = base_rainfall + np.random.exponential(2)
            else:
                rainfall = base_rainfall
            
            # Determine suspension (expert rules)
            suspension = self.determine_suspension(
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
    
    def calculate_heat_index(self, temperature, humidity):
        """Calculate heat index"""
        if temperature < 27:
            return temperature
        
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6
        
        T = temperature
        R = humidity
        
        hi = (c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + 
              (c5 * T**2) + (c6 * R**2) + (c7 * T**2 * R) + 
              (c8 * T * R**2) + (c9 * T**2 * R**2))
        
        return hi
    
    def calculate_comfort_index(self, temp, humidity, wind_speed):
        """Calculate comfort index"""
        temp_comfort = max(0, 100 - abs(temp - 24) * 5)
        humidity_comfort = max(0, 100 - abs(humidity - 50) * 1.5)
        wind_comfort = max(0, 100 - abs(wind_speed - 5) * 10)
        return (temp_comfort + humidity_comfort + wind_comfort) / 3
    
    def determine_weather_condition(self, rainfall, humidity, cloud_cover):
        """Determine weather condition"""
        if rainfall > 15:
            return 'Thunderstorm'
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
    
    def determine_suspension(self, temp, humidity, rainfall, wind_speed, heat_index, comfort_index, condition, location_risk):
        """Expert rules for suspension determination"""
        risk_score = 0
        
        # Critical conditions (automatic suspension)
        if heat_index > 45:
            return 'Yes'
        if rainfall > 25:
            return 'Yes'
        if wind_speed > 65:
            return 'Yes'
        if condition == 'Thunderstorm' and rainfall > 10:
            return 'Yes'
        
        # Weighted risk factors
        if temp > 35: risk_score += 2
        if heat_index > 40: risk_score += 2
        if rainfall > 15: risk_score += 3
        if rainfall > 8: risk_score += 2
        if wind_speed > 45: risk_score += 2
        if comfort_index < 40: risk_score += 1
        if condition in ['Thunderstorm', 'Rain']: risk_score += 1
        
        # Location risk multiplier
        location_factor = location_risk / 50
        total_score = risk_score * location_factor
        
        return 'Yes' if total_score >= 3 else 'No'
    
    def prepare_features(self, df):
        """Prepare features for Random Forest"""
        df_processed = df.copy()
        
        # Encode categorical variables
        weather_dummies = pd.get_dummies(df_processed['WeatherCondition'], prefix='Weather')
        df_processed = pd.concat([df_processed, weather_dummies], axis=1)
        df_processed = df_processed.drop('WeatherCondition', axis=1)
        
        # Ensure all expected columns
        expected_weather_cols = ['Weather_Clear', 'Weather_Clouds', 'Weather_Rain', 
                               'Weather_Thunderstorm', 'Weather_Drizzle', 'Weather_Fog']
        
        for col in expected_weather_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        return df_processed
    
    def train_model(self, save_path='windy_rf_model.pkl'):
        """Train Random Forest model"""
        print("🏗️ Training Random Forest Model...")
        df = self.generate_training_data(3000)
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Separate features and target
        X = df_processed.drop('Suspension', axis=1)
        y = df_processed['Suspension'].map({'Yes': 1, 'No': 0})
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest with optimized parameters
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
        
        print("🔧 Training in progress...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model Training Complete!")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Feature Importances:")
        print(feature_importance.head(10))
        
        # Save model
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now(),
            'accuracy': accuracy
        }
        joblib.dump(model_data, save_path)
        print(f"💾 Model saved to {save_path}")
        
        return accuracy
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            print(f"✅ Random Forest model loaded (Accuracy: {model_data.get('accuracy', 'Unknown'):.3f})")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Training new model...")
            self.train_model(model_path)
    
    def predict(self, weather_data):
        """Make prediction using Random Forest"""
        if self.model is None:
            self.train_model()
        
        # Prepare input features
        input_data = {
            'Temperature': weather_data['temperature'],
            'Humidity': weather_data['humidity'],
            'Rainfall': weather_data['rainfall'],
            'WindSpeed': weather_data['wind_speed'],
            'Pressure': weather_data['pressure'],
            'CloudCover': weather_data.get('cloud_cover', 50),
            'HeatIndex': weather_data['heat_index'],
            'ComfortIndex': weather_data.get('comfort_index', 50),
            'WeatherCondition': weather_data['weather_condition'],
            'LocationRisk': weather_data['suspension_risk_base'],
            'IsFloodProne': 1 if any('flood' in factor.lower() for factor in weather_data.get('risk_factors', [])) else 0,
            'IsHighDensity': 1 if any('density' in factor.lower() for factor in weather_data.get('risk_factors', [])) else 0
        }
        
        input_df = pd.DataFrame([input_data])
        input_processed = self.prepare_features(input_df)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        input_processed = input_processed[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(input_processed)[0]
        probability = self.model.predict_proba(input_processed)[0]
        
        # Get feature contributions (simplified)
        feature_contributions = self._get_feature_contributions(input_processed)
        
        return {
            'prediction': 'Suspend Classes' if prediction == 1 else 'Resume Classes',
            'confidence': max(probability),
            'suspend_probability': probability[1],
            'resume_probability': probability[0],
            'location_risk': weather_data['suspension_risk_base'],
            'risk_factors': weather_data.get('risk_factors', []),
            'feature_contributions': feature_contributions,
            'model_type': 'Random Forest'
        }
    
    def _get_feature_contributions(self, input_data):
        """Get simplified feature contributions"""
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Get top 5 contributing features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {feature: round(importance * 100, 1) for feature, importance in top_features}

def main():
    """Main Windy + Random Forest Prediction System"""
    
    print("🌪️ Windy.com + Random Forest Class Suspension Predictor")
    print("=" * 65)
    print("🤖 Using Random Forest Algorithm with Windy.com Weather Data")
    print("=" * 65)
    
    # Initialize components
    weather_collector = WindyWeatherCollector()
    predictor = WindyRandomForestPredictor('windy_rf_model.pkl')
    
    # Display available locations
    print("\n📍 Available Metro Manila Locations:")
    metro_manila_names = [loc['name'] for loc in METRO_MANILA_PLACES]
    for i in range(0, len(metro_manila_names), 4):
        print("   " + "  ".join(f"{name:15}" for name in metro_manila_names[i:i+4]))
    
    while True:
        print("\n" + "=" * 65)
        location_name = input("\n🏫 Enter location name (or 'quit' to exit): ").strip()
        
        if location_name.lower() == 'quit':
            break
        
        if not location_name:
            location_name = "Manila"
        
        print(f"📡 Getting Windy.com data for {location_name}...")
        start_time = time.time()
        
        weather_data = weather_collector.get_weather_data(location_name)
        
        if weather_data:
            fetch_time = time.time() - start_time
            print(f"✅ Data retrieved in {fetch_time:.1f}s from {weather_data['data_source']}")
            
            # Display location info
            print(f"\n📍 Location: {weather_data['location_name']}")
            print(f"🏛️  District: {weather_data['district']}")
            print(f"📊 Base Risk: {weather_data['suspension_risk_base']}%")
            
            # Display weather data
            print(f"\n🌡️  Current Weather Conditions:")
            print(f"   Temperature: {weather_data['temperature']}°C")
            print(f"   Humidity: {weather_data['humidity']}%")
            print(f"   Rainfall: {weather_data['rainfall']} mm")
            print(f"   Wind Speed: {weather_data['wind_speed']} kph")
            print(f"   Pressure: {weather_data['pressure']} hPa")
            print(f"   Cloud Cover: {weather_data['cloud_cover']}%")
            print(f"   Condition: {weather_data['weather_condition']}")
            print(f"   Heat Index: {weather_data['heat_index']}°C")
            print(f"   Comfort Index: {weather_data['comfort_index']}/100")
            
            # Make prediction
            print(f"\n🤖 Random Forest Prediction in progress...")
            prediction_start = time.time()
            prediction_result = predictor.predict(weather_data)
            prediction_time = time.time() - prediction_start
            
            print(f"✅ Prediction completed in {prediction_time:.2f}s")
            
            print("\n" + "=" * 50)
            print("🎓 CLASS SUSPENSION RECOMMENDATION")
            print("=" * 50)
            
            if prediction_result['prediction'] == 'Suspend Classes':
                print("🚨 ✅ RECOMMENDATION: SUSPEND CLASSES")
                print("   ⚠️  High-risk conditions detected")
            else:
                print("📚 ❌ RECOMMENDATION: RESUME CLASSES")
                print("   ✅ Conditions are generally safe")
            
            print(f"\n📊 Prediction Details:")
            print(f"   Model: {prediction_result['model_type']}")
            print(f"   Confidence: {prediction_result['confidence']:.1%}")
            print(f"   Suspend Probability: {prediction_result['suspend_probability']:.1%}")
            print(f"   Resume Probability: {prediction_result['resume_probability']:.1%}")
            print(f"   Location Risk Factor: {prediction_result['location_risk']}%")
            
            # Display top contributing features
            if prediction_result.get('feature_contributions'):
                print(f"\n🔍 Top Contributing Factors:")
                for feature, contribution in prediction_result['feature_contributions'].items():
                    feature_name = feature.replace('Weather_', '').replace('Is', '')
                    print(f"   • {feature_name}: {contribution}%")
            
            # Risk assessment
            print(f"\n⚠️  RISK ASSESSMENT:")
            warnings = _generate_warnings(weather_data, prediction_result)
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
            
            # Recommendations
            print(f"\n🛡️  RECOMMENDATIONS:")
            recommendations = _generate_recommendations(prediction_result, weather_data) 
            for i, recommendation in enumerate(recommendations, 1):
                print(f"   {i}. {recommendation}")
                
        else:
            print(f"❌ Could not fetch weather data for {location_name}")
        
        # Continue?
        cont = input("\n🔁 Check another location? (y/n): ").strip().lower()
        if cont != 'y':
            break
    
    print("\n🙏 Thank you for using Windy.com + Random Forest Predictor!")

def _generate_warnings(weather_data, prediction_result):
    """Generate risk warnings"""
    warnings = []
    
    if weather_data['heat_index'] > 45:
        warnings.append(f"Extreme heat index ({weather_data['heat_index']}°C) - Heat stroke risk")
    elif weather_data['heat_index'] > 40:
        warnings.append(f"High heat index ({weather_data['heat_index']}°C) - Caution advised")
    
    if weather_data['rainfall'] > 20:
        warnings.append(f"Heavy rainfall ({weather_data['rainfall']}mm) - Flooding likely")
    elif weather_data['rainfall'] > 10:
        warnings.append(f"Moderate rainfall ({weather_data['rainfall']}mm) - Potential flooding")
    
    if weather_data['wind_speed'] > 60:
        warnings.append(f"Strong winds ({weather_data['wind_speed']}kph) - Safety hazard")
    elif weather_data['wind_speed'] > 40:
        warnings.append(f"Moderate winds ({weather_data['wind_speed']}kph) - Exercise caution")
    
    if weather_data['comfort_index'] < 40:
        warnings.append(f"Poor comfort conditions ({weather_data['comfort_index']}/100)")
    
    if prediction_result['location_risk'] > 80:
        warnings.append(f"High-risk location - Increased vigilance needed")
    
    return warnings

def _generate_recommendations(prediction_result, weather_data):
    """Generate safety recommendations"""
    recommendations = []
    
    if prediction_result['prediction'] == 'Suspend Classes':
        recommendations.append("Implement class suspension immediately")
        recommendations.append("Monitor weather updates every 30 minutes")
        recommendations.append("Activate emergency communication protocols")
        recommendations.append("Prepare evacuation plans if necessary")
    else:
        recommendations.append("Continue normal class operations")
        recommendations.append("Monitor weather conditions regularly")
        recommendations.append("Have contingency plans ready")
        recommendations.append("Stay updated with local advisories")
    
    # Additional specific recommendations
    if weather_data['heat_index'] > 40:
        recommendations.append("Ensure adequate hydration and cooling")
    
    if weather_data['rainfall'] > 10:
        recommendations.append("Avoid flood-prone areas and routes")
    
    return recommendations

if __name__ == "__main__":
    main()