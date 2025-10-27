from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
import traceback


from services.weather_service import WeatherService
from services.ml_prediction_service import MLPredictionService
from data.locations import METRO_MANILA_PLACES, get_location_by_name, get_all_locations
from models.database import db_manager

app = Flask(__name__)
CORS(app)  


weather_service = WeatherService()
ml_prediction_service = MLPredictionService()


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'weather_service': 'active',
            'ml_prediction_service': 'active'
        }
    })


@app.route('/api/locations')
def get_locations():
    """Get all locations with dynamic risk calculations based on current weather"""
    try:
        from data.locations import METRO_MANILA_PLACES
        
        # Get base location data
        base_locations = get_all_locations()
        
        # Calculate dynamic risk for each location
        locations_with_risk = []
        for location in base_locations:
            try:
                # Fetch current weather for this location
                weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
                
                if weather_data:
                    # Calculate dynamic risk percentage using ML model
                    risk_percentage = ml_prediction_service._calculate_risk_percentage(weather_data, location)
                    
                    # Generate dynamic risk factors based on weather
                    risk_factors = ml_prediction_service._generate_dynamic_risk_factors(weather_data, location)
                    
                    # Create enhanced location data
                    enhanced_location = location.copy()
                    enhanced_location['suspensionRisk'] = round(risk_percentage)
                    enhanced_location['riskFactors'] = risk_factors
                    enhanced_location['weatherUpdated'] = datetime.now().isoformat()
                    
                    locations_with_risk.append(enhanced_location)
                else:
                    # Fallback to static data if weather fetch fails
                    locations_with_risk.append(location)
                    
            except Exception as e:
                print(f"Error calculating risk for {location['name']}: {e}")
                # Fallback to static data
                locations_with_risk.append(location)
        
        return jsonify({
            'locations': locations_with_risk,
            'total': len(locations_with_risk),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locations/<location_name>')
def get_location_details(location_name):

    try:
        location = get_location_by_name(location_name)
        if not location:
            return jsonify({'error': f'Location {location_name} not found'}), 404
        
        return jsonify({
            'location': location,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/current')
def get_current_weather():

    try:
        city_name = request.args.get('city', 'Manila')
        

        location = get_location_by_name(city_name)
        if not location:
            return jsonify({'error': f'City {city_name} not found'}), 404
        
        weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
        if not weather_data:
            return jsonify({'error': 'Could not fetch weather data'}), 500
        
        weather_data['location'] = {
            'name': location['name'],
            'lat': location['lat'],
            'lon': location['lon'],
            'district': location['district'],
            'type': location['type'],
            'suspensionRisk': location['suspensionRisk'],
            'riskFactors': location['riskFactors']
        }
        
        return jsonify(weather_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/forecast')
def get_weather_forecast():

    try:
        city_name = request.args.get('city', 'Manila')
        
        location = get_location_by_name(city_name)
        if not location:
            return jsonify({'error': f'City {city_name} not found'}), 404
        
        forecast_data = weather_service.get_forecast_weather(location['lat'], location['lon'])
        if not forecast_data:
            return jsonify({'error': 'Could not fetch forecast data'}), 500
        
        forecast_data['location'] = {
            'name': location['name'],
            'lat': location['lat'],
            'lon': location['lon']
        }
        
        return jsonify(forecast_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/hourly')
def get_hourly_forecast():
    """Get 24-hour hourly forecast"""
    try:
        city_name = request.args.get('city', 'Manila')
        
        location = get_location_by_name(city_name)
        if not location:
            return jsonify({'error': f'City {city_name} not found'}), 404
        
        hourly_data = weather_service.get_hourly_forecast(location['lat'], location['lon'])
        if not hourly_data:
            return jsonify({'error': 'Could not fetch hourly data'}), 500
        
        hourly_data['location'] = {
            'name': location['name'],
            'lat': location['lat'],
            'lon': location['lon']
        }
        
        return jsonify(hourly_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/suspension', methods=['POST'])
def predict_class_suspension():

    try:
        data = request.json
        city_name = data.get('city', 'Manila')
        
      
        location = get_location_by_name(city_name)
        if not location:
            return jsonify({'error': f'City {city_name} not found'}), 404
        
      
        weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
        if not weather_data:
            return jsonify({'error': 'Could not fetch weather data'}), 500
        
        # Make ML prediction with logging
        prediction = ml_prediction_service.predict_suspension(
            weather_data, location,
            api_endpoint='/api/predict/suspension',
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr,
            session_id=request.headers.get('X-Session-ID', '')
        )
        
        prediction['location'] = {
            'name': location['name'],
            'district': location['district'],
            'suspensionRisk': location['suspensionRisk'],
            'riskFactors': location['riskFactors']
        }
        
        prediction['weather_summary'] = {
            'temperature': weather_data['current']['temp_c'],
            'humidity': weather_data['current']['humidity'],
            'rainfall': weather_data['current']['precip_mm'],
            'wind_speed': weather_data['current']['wind_kph'],
            'condition': weather_data['current']['condition']['text']
        }
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/bulk', methods=['POST'])
def bulk_predict_suspension():
    try:
        data = request.json
        cities = data.get('cities', ['Manila', 'Quezon City', 'Makati'])
        
        results = []
        for city_name in cities[:10]:  
            try:
                location = get_location_by_name(city_name)
                if not location:
                    continue
                
                weather_data = weather_service.get_current_weather(location['lat'], location['lon'])
                if not weather_data:
                    continue
                
                prediction = ml_prediction_service.predict_suspension(
                    weather_data, location,
                    api_endpoint='/api/predict/bulk',
                    user_agent=request.headers.get('User-Agent', ''),
                    ip_address=request.remote_addr,
                    session_id=request.headers.get('X-Session-ID', '')
                )
                
                results.append({
                    'city': city_name,
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'risk_level': prediction['risk_level'],
                    'risk_percentage': prediction['risk_percentage'],
                    'location_risk': location['suspensionRisk'],
                    'weather_summary': {
                        'temperature': weather_data['current']['temp_c'],
                        'rainfall': weather_data['current']['precip_mm'],
                        'condition': weather_data['current']['condition']['text']
                    }
                })
                
       
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing {city_name}: {e}")
                continue
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk/top-lgus')
def get_top_risk_lgus():
    try:
  
        high_risk_locations = [loc for loc in METRO_MANILA_PLACES if loc['suspensionRisk'] > 70]
        
      
        weather_data_list = weather_service.get_multiple_locations_weather(high_risk_locations)
        

        results = []
        for weather_data in weather_data_list:
            location = weather_data['location']
            
         
            prediction = ml_prediction_service.predict_suspension(
                weather_data, location,
                api_endpoint='/api/risk/top-lgus',
                user_agent=request.headers.get('User-Agent', ''),
                ip_address=request.remote_addr,
                session_id=request.headers.get('X-Session-ID', '')
            )
            
        
            dynamic_risk = prediction['risk_percentage']
            
            results.append({
                'name': location['name'],
                'district': location['district'],
                'suspensionRisk': location['suspensionRisk'],
                'dynamicRisk': dynamic_risk,
                'weatherCondition': weather_data['current']['condition']['text'],
                'temperature': weather_data['current']['temp_c'],
                'rainfall': weather_data['current']['precip_mm'],
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'risk_level': prediction['risk_level']
            })
        
   
        results.sort(key=lambda x: x['dynamicRisk'], reverse=True)
        top_5 = results[:5]
        
        return jsonify({
            'top_risk_lgus': top_5,
            'total_analyzed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/complete')
def get_complete_weather_data():
    """Get complete weather data including current, forecast, hourly, and prediction"""
    try:
        city_name = request.args.get('city', 'Manila')
        
       
        location = get_location_by_name(city_name)
        if not location:
            return jsonify({'error': f'City {city_name} not found'}), 404
        
  
        current_weather = weather_service.get_current_weather(location['lat'], location['lon'])
        if not current_weather:
            return jsonify({'error': 'Could not fetch weather data'}), 500
        

        prediction = ml_prediction_service.predict_suspension(
            current_weather, location,
            api_endpoint='/api/weather/complete',
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr,
            session_id=request.headers.get('X-Session-ID', '')
        )
        

        response = {
            'location': {
                'name': location['name'],
                'lat': location['lat'],
                'lon': location['lon'],
                'district': location['district'],
                'type': location['type'],
                'suspensionRisk': location['suspensionRisk'],
                'riskFactors': location['riskFactors']
            },
            'current': current_weather['current'],
            'forecast': current_weather['forecast'],
            'hourly': current_weather['hourly'],
            'prediction': prediction,
            'metadata': current_weather['metadata']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Database endpoints for AI logs
@app.route('/api/logs/predictions')
def get_prediction_logs():
    """Get AI prediction logs with optional filters"""
    try:
        limit = request.args.get('limit', 100, type=int)
        location_name = request.args.get('location')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        success_only = request.args.get('success_only', 'true').lower() == 'true'
        
        # Parse dates if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        logs = db_manager.get_prediction_logs(
            limit=limit,
            location_name=location_name,
            start_date=start_dt,
            end_date=end_dt,
            success_only=success_only
        )
        
        return jsonify({
            'logs': logs,
            'total': len(logs),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/statistics')
def get_prediction_statistics():
    """Get AI prediction statistics"""
    try:
        stats = db_manager.get_prediction_statistics()
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/cleanup', methods=['POST'])
def cleanup_old_logs():
    """Clean up old prediction logs"""
    try:
        data = request.json or {}
        days_to_keep = data.get('days_to_keep', 30)
        
        deleted_count = db_manager.cleanup_old_logs(days_to_keep)
        
        return jsonify({
            'message': f'Cleaned up {deleted_count} old logs',
            'deleted_count': deleted_count,
            'days_kept': days_to_keep,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)