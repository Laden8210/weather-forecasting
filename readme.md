# Weather Prediction Backend

A comprehensive Flask backend API that provides weather data and ML-powered class suspension predictions for Metro Manila LGUs.

## 🚀 Features

- **Real-time Weather Data**: Integration with Open-Meteo API for accurate weather information
- **ML Predictions**: Random Forest model for class suspension predictions
- **Multi-location Support**: Weather data for all Metro Manila cities
- **Risk Assessment**: Dynamic risk calculation based on weather and location factors
- **RESTful API**: Clean API endpoints for frontend integration

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🛠️ Installation

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Server

### Option 1: Using the start script (Recommended)
```bash
# From project root
python start_backend.py
```

### Option 2: Manual start
```bash
# From backend directory
cd backend
python app.py
```

The server will start on `http://localhost:5000`

## 📡 API Endpoints

### Health Check
- `GET /health` - Server health status

### Locations
- `GET /api/locations` - Get all Metro Manila locations
- `GET /api/locations/{name}` - Get specific location details

### Weather Data
- `GET /api/weather/current?city={name}` - Current weather for a city
- `GET /api/weather/forecast?city={name}` - 3-day weather forecast
- `GET /api/weather/hourly?city={name}` - 24-hour hourly forecast
- `GET /api/weather/complete?city={name}` - Complete weather data with prediction

### ML Predictions
- `POST /api/predict/suspension` - Predict class suspension for a city
- `POST /api/predict/bulk` - Bulk predictions for multiple cities

### Risk Assessment
- `GET /api/risk/top-lgus` - Top 5 high-risk LGUs with current data

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the backend directory:
```env
FLASK_ENV=development
FLASK_DEBUG=True
API_TIMEOUT=10
```

### Model Configuration
The ML model is automatically loaded from `windy_rf_model.pkl`. If the model doesn't exist, a new one will be trained automatically.

## 📊 Data Sources

- **Weather Data**: Open-Meteo API (free, no API key required)
- **Location Data**: Metro Manila cities database
- **ML Model**: Random Forest trained on historical weather patterns

## 🧪 Testing the API

### Test health endpoint:
```bash
curl http://localhost:5000/health
```

### Test weather data:
```bash
curl "http://localhost:5000/api/weather/complete?city=Manila"
```

### Test ML prediction:
```bash
curl -X POST http://localhost:5000/api/predict/suspension \
  -H "Content-Type: application/json" \
  -d '{"city": "Manila"}'
```

## 🏗️ Architecture

```
backend/
├── app.py                 # Main Flask application
├── services/
│   ├── weather_service.py      # Weather data service
│   └── ml_prediction_service.py # ML prediction service
├── data/
│   └── locations.py           # Location database
├── windy_rf_model.pkl         # Trained ML model
└── requirements.txt           # Python dependencies
```

## 🔍 Troubleshooting

### Common Issues

1. **Port 5000 already in use**:
   ```bash
   # Kill process using port 5000
   lsof -ti:5000 | xargs kill -9
   ```

2. **Model loading errors**:
   - The system will automatically train a new model if loading fails
   - Check that scikit-learn is properly installed

3. **Weather API timeouts**:
   - Check internet connection
   - Open-Meteo API is free but may have rate limits

4. **CORS errors**:
   - CORS is enabled for all origins
   - If issues persist, check browser console for specific errors

### Logs
The server logs all requests and errors to the console. Check the terminal output for debugging information.

## 📈 Performance

- **Weather API**: ~1-2 seconds per request
- **ML Prediction**: ~0.1-0.5 seconds per prediction
- **Bulk Operations**: Rate limited to prevent API overload

## 🔒 Security

- CORS enabled for development
- Input validation on all endpoints
- Error handling prevents information leakage
- No sensitive data stored

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.