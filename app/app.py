from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import save_prediction, get_all_records, get_statistics, delete_record
from chart_generator import (
    generate_disease_chart,
    generate_age_chart,
    generate_seasonal_chart,
    generate_gender_chart,
    generate_medicine_chart,
    generate_feature_importance,
    generate_seasonal_heatmap
)

app = Flask(__name__)

# Load models and encoders
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, 'models')
PLOTS = os.path.join(BASE, 'plots')

# Check if models exist
if not os.path.exists(os.path.join(MODELS, 'random_forest.pkl')):
    print("\n" + "=" * 60)
    print("❌ ERROR: Models not found!")
    print("   Please run: python analysis.py")
    print("=" * 60 + "\n")
    exit(1)

print("\n[✓] Loading models...")
model = joblib.load(os.path.join(MODELS, 'random_forest.pkl'))
le_gender = joblib.load(os.path.join(MODELS, 'le_gender.pkl'))
le_medicine = joblib.load(os.path.join(MODELS, 'le_medicine.pkl'))
le_disease = joblib.load(os.path.join(MODELS, 'le_disease.pkl'))

# Load scaler if exists, otherwise create dummy
scaler_path = os.path.join(MODELS, 'scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    print("   Warning: scaler.pkl not found, using new scaler")

print("[✓] Models loaded successfully")

# Frequency mapping
freq_map = {
    'Once daily': 1,
    'Twice daily': 2,
    'Three times daily': 3,
    'Once weekly': 0.14,
    'As needed': 0.5
}

# ========== STATIC ROUTES ==========

# Route to serve plot images
@app.route('/plots/<path:filename>')
def serve_plots(filename):
    return send_from_directory(PLOTS, filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/realtime-dashboard')
def realtime_dashboard():
    return render_template('realtime_dashboard.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/records')
def records():
    return render_template('records.html')

# ✅ CORRECTED: Changed from '/seasonal-live' to '/seasonal'
@app.route('/seasonal')
def seasonal():
    return render_template('seasonal_live.html')

# ========== REAL-TIME CHART APIs ==========

@app.route('/api/charts/disease')
def api_chart_disease():
    chart = generate_disease_chart()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/age')
def api_chart_age():
    chart = generate_age_chart()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/seasonal')
def api_chart_seasonal():
    chart = generate_seasonal_chart()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/gender')
def api_chart_gender():
    chart = generate_gender_chart()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/medicine')
def api_chart_medicine():
    chart = generate_medicine_chart()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/features')
def api_chart_features():
    chart = generate_feature_importance()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

@app.route('/api/charts/seasonal-heatmap')
def api_chart_seasonal_heatmap():
    chart = generate_seasonal_heatmap()
    if chart:
        return jsonify({'success': True, 'chart': chart})
    return jsonify({'success': False, 'error': 'No data available'})

# ========== PREDICTION API ==========

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Prepare features
        gender_enc = le_gender.transform([data['gender']])[0]
        medicine_enc = le_medicine.transform([data['medicine']])[0]
        freq_enc = freq_map.get(data['frequency'], 1)
        
        features = np.array([[
            float(data['age']),
            gender_enc,
            medicine_enc,
            float(data['dosage_mg']),
            freq_enc,
            int(data['month'])
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction_enc = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities) * 100
        disease = le_disease.inverse_transform([prediction_enc])[0]
        
        # Save to database
        record_data = {
            'patient_name': data.get('patient_name', 'Anonymous'),
            'age': data['age'],
            'gender': data['gender'],
            'disease': disease,
            'medicine': data['medicine'],
            'dosage_mg': data['dosage_mg'],
            'frequency': data['frequency'],
            'month': data['month'],
            'confidence': confidence
        }
        save_prediction(record_data)
        
        return jsonify({
            'success': True,
            'disease': disease,
            'confidence': round(confidence, 2),
            'probabilities': {
                le_disease.inverse_transform([i])[0]: round(probabilities[i] * 100, 2)
                for i in range(len(probabilities))
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ========== RECORDS APIs ==========

@app.route('/api/records')
def api_records():
    df = get_all_records()
    records = df.to_dict('records') if not df.empty else []
    return jsonify(records)

@app.route('/api/stats')
def api_stats():
    stats = get_statistics()
    return jsonify(stats)

@app.route('/api/records/<int:record_id>', methods=['DELETE'])
def delete_record_route(record_id):
    try:
        delete_record(record_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ========== INSIGHTS API (for seasonal page) ==========

@app.route('/api/insights')
def api_insights():
    df = get_all_records()
    insights = []
    
    if not df.empty:
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Spring'
            elif month in [6, 7, 8]: return 'Monsoon'
            else: return 'Autumn'
        
        df['season'] = df['month'].apply(get_season)
        
        # Season insights
        for season in ['Monsoon', 'Winter', 'Spring', 'Autumn']:
            season_data = df[df['season'] == season]
            if len(season_data) > 0:
                top = season_data['disease'].mode()[0] if not season_data.empty else 'None'
                insights.append(f"{season}: {top} is most common ({len(season_data)} cases)")
        
        # Age insights
        elderly = df[df['age'] > 60]
        if len(elderly) > 0:
            top = elderly['disease'].mode()[0] if not elderly.empty else 'None'
            insights.append(f"Elderly (60+): Most affected by {top}")
        
        children = df[df['age'] < 12]
        if len(children) > 0:
            top = children['disease'].mode()[0] if not children.empty else 'None'
            insights.append(f"Children (under 12): Most affected by {top}")
    
    if not insights:
        insights = ["Make predictions to see personalized insights!"]
    
    return jsonify(insights)

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  🏥 Prescription Analyzer Started")
    print("  📍 http://localhost:5000")
    print("  📍 Real-Time Dashboard: http://localhost:5000/realtime-dashboard")
    print("  📍 Seasonal Trends: http://localhost:5000/seasonal")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)