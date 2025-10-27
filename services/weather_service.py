import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class WeatherService:
    """Service for fetching weather data from Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.timeout = 10
        
    def get_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather data for coordinates"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m',
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m',
                'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant',
                'timezone': 'Asia/Manila',
                'forecast_days': 3
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return self._process_current_weather(data)
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return None
    
    def get_forecast_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get 3-day forecast weather data"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant',
                'timezone': 'Asia/Manila',
                'forecast_days': 3
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return self._process_forecast_weather(data)
            
        except Exception as e:
            print(f"Forecast API error: {e}")
            return None
    
    def get_hourly_forecast(self, lat: float, lon: float) -> Optional[Dict]:
        """Get 24-hour hourly forecast"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m',
                'timezone': 'Asia/Manila',
                'forecast_hours': 24
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return self._process_hourly_forecast(data)
            
        except Exception as e:
            print(f"Hourly forecast API error: {e}")
            return None
    
    def get_multiple_locations_weather(self, locations: List[Dict]) -> List[Dict]:
        """Get weather data for multiple locations"""
        results = []
        
        for location in locations:
            try:
                weather_data = self.get_current_weather(location['lat'], location['lon'])
                if weather_data:
                    weather_data['location'] = location
                    results.append(weather_data)
                
              
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching weather for {location['name']}: {e}")
                continue
        
        return results
    
    def _process_current_weather(self, data: Dict) -> Dict:
        """Process current weather data from API response"""
        current = data.get('current', {})
        hourly = data.get('hourly', {})
        
        
        temp_c = current.get('temperature_2m', 0)
        humidity = current.get('relative_humidity_2m', 0)
        wind_speed = current.get('wind_speed_10m', 0)
        
        
        heat_index = self._calculate_heat_index(temp_c, humidity)
        
        
        comfort_index = self._calculate_comfort_index(temp_c, humidity, wind_speed)
        
        return {
            'current': {
                'temp_c': temp_c,
                'feelslike_c': heat_index,
                'condition': {
                    'text': self._get_weather_description(current.get('weather_code', 0)),
                    'code': current.get('weather_code', 0)
                },
                'wind_kph': wind_speed,
                'gust_kph': current.get('wind_gusts_10m', wind_speed * 1.5),
                'precip_mm': current.get('precipitation', 0),
                'humidity': humidity,
                'pressure_mb': current.get('pressure_msl', 1013),
                'cloud_cover': current.get('cloud_cover', 0),
                'wind_direction': current.get('wind_direction_10m', 0)
            },
            'forecast': self._process_forecast_weather(data),
            'hourly': self._process_hourly_forecast(data),
            'metadata': {
                'heat_index': heat_index,
                'comfort_index': comfort_index,
                'last_updated': datetime.now().isoformat(),
                'data_source': 'Open-Meteo'
            }
        }
    
    def _process_forecast_weather(self, data: Dict) -> Dict:
        
        daily = data.get('daily', {})
        
        if not daily:
            return {'forecastday': []}
        
        forecast_days = []
        dates = daily.get('time', [])
        
        for i in range(min(3, len(dates))):     
            day_data = {
                'date': dates[i],
                'day': {
                    'maxtemp_c': daily.get('temperature_2m_max', [0] * len(dates))[i],
                    'mintemp_c': daily.get('temperature_2m_min', [0] * len(dates))[i],
                    'totalprecip_mm': daily.get('precipitation_sum', [0] * len(dates))[i],
                    'maxwind_kph': daily.get('wind_speed_10m_max', [0] * len(dates))[i],
                    'condition': {
                        'text': self._get_weather_description(daily.get('weather_code', [0] * len(dates))[i]),
                        'code': daily.get('weather_code', [0] * len(dates))[i]
                    }
                }
            }
            forecast_days.append(day_data)
        
        return {'forecastday': forecast_days}
    
    def _process_hourly_forecast(self, data: Dict) -> Dict:
        """Process hourly forecast data"""
        hourly = data.get('hourly', {})
        
        if not hourly:
            return {'hour': []}
        
        hourly_data = []
        times = hourly.get('time', [])
        
        for i in range(min(24, len(times))):    
            hour_data = {
                'time': times[i],
                'temp_c': hourly.get('temperature_2m', [0] * len(times))[i],
                'condition': {
                    'text': self._get_weather_description(hourly.get('weather_code', [0] * len(times))[i]),
                    'code': hourly.get('weather_code', [0] * len(times))[i]
                },
                'precip_mm': hourly.get('precipitation', [0] * len(times))[i],
                'wind_kph': hourly.get('wind_speed_10m', [0] * len(times))[i],
                'humidity': hourly.get('relative_humidity_2m', [0] * len(times))[i],
                'cloud': hourly.get('cloud_cover', [0] * len(times))[i]
            }
            hourly_data.append(hour_data)
        
        return {'hour': hourly_data}
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index using NOAA formula"""
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
        """Calculate comfort index (0-100, higher is better)"""
            
        temp_comfort = max(0, 100 - abs(temp_c - 24) * 5)
        
        
        humidity_comfort = max(0, 100 - abs(humidity - 50) * 1.5)
        
        
        wind_comfort = max(0, 100 - abs(wind_speed - 5) * 10)
        
        return (temp_comfort + humidity_comfort + wind_comfort) / 3
    
    def _get_weather_description(self, weather_code: int) -> str:
        """Convert weather code to description"""
        weather_codes = {
            0: "Clear",
            1: "Clear",
            2: "Clouds",
            3: "Clouds",
            45: "Fog",
            48: "Fog",
            51: "Drizzle",
            53: "Drizzle",
            55: "Drizzle",
            56: "Drizzle",
            57: "Drizzle",
            61: "Rain",
            63: "Rain",
            65: "Rain",
            66: "Rain",
            67: "Rain",
            71: "Snow",
            73: "Snow",
            75: "Snow",
            77: "Snow",
            80: "Rain",
            81: "Rain",
            82: "Rain",
            85: "Snow",
            86: "Snow",
            95: "Thunderstorm",
            96: "Thunderstorm",
            99: "Thunderstorm"
        }
        
        return weather_codes.get(weather_code, "Clear")
