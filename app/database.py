import sqlite3
import pandas as pd
import os
from datetime import datetime

# ========== FIX FOR DEPLOYMENT ==========
# Use /tmp for Render, local for development

# Get database path
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if running on Render (cloud) or local
if os.path.exists('/tmp'):
    # Running on Render (cloud) - use temporary storage
    DB_PATH = os.path.join('/tmp', 'prescriptions.db')
    print("[✓] Running on cloud - using /tmp/prescriptions.db")
else:
    # Running locally - use data folder
    DB_PATH = os.path.join(BASE, 'data', 'prescriptions.db')
    print("[✓] Running locally - using data/prescriptions.db")

# Create directory if needed
os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)

def init_db():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            disease TEXT,
            medicine TEXT,
            dosage_mg REAL,
            frequency TEXT,
            month INTEGER,
            season TEXT,
            prediction_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[✓] Database initialized at:", DB_PATH)

def save_prediction(data):
    """Save a prediction record to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Determine season from month
    month = int(data['month'])
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Monsoon'
    else:
        season = 'Autumn'
    
    cursor.execute('''
        INSERT INTO prescriptions (patient_name, age, gender, disease, medicine, 
                                   dosage_mg, frequency, month, season, prediction_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['patient_name'], data['age'], data['gender'], data['disease'],
        data['medicine'], data['dosage_mg'], data['frequency'], 
        data['month'], season, data['confidence']
    ))
    
    conn.commit()
    conn.close()
    print("[✓] Prediction saved to database")

def get_all_records():
    """Get all prescription records"""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM prescriptions ORDER BY created_at DESC", conn)
    conn.close()
    
    # Convert to native types for JSON serialization
    if not df.empty:
        for col in df.columns:
            if df[col].dtype == 'int64':
                df[col] = df[col].astype(int)
            elif df[col].dtype == 'float64':
                df[col] = df[col].astype(float)
    
    return df

def get_statistics():
    """Get summary statistics from database"""
    if not os.path.exists(DB_PATH):
        return {
            'total': 0,
            'top_disease': 'N/A',
            'top_disease_count': 0,
            'top_medicine': 'N/A',
            'top_medicine_count': 0,
            'seasonal': []
        }
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        total = pd.read_sql_query("SELECT COUNT(*) as count FROM prescriptions", conn).iloc[0]['count']
        total = int(total) if not pd.isna(total) else 0
    except:
        total = 0
    
    try:
        top_disease = pd.read_sql_query("""
            SELECT disease, COUNT(*) as count 
            FROM prescriptions 
            GROUP BY disease 
            ORDER BY count DESC 
            LIMIT 1
        """, conn)
        if len(top_disease) > 0:
            top_disease_name = str(top_disease.iloc[0]['disease'])
            top_disease_count = int(top_disease.iloc[0]['count'])
        else:
            top_disease_name = 'N/A'
            top_disease_count = 0
    except:
        top_disease_name = 'N/A'
        top_disease_count = 0
    
    try:
        top_medicine = pd.read_sql_query("""
            SELECT medicine, COUNT(*) as count 
            FROM prescriptions 
            GROUP BY medicine 
            ORDER BY count DESC 
            LIMIT 1
        """, conn)
        if len(top_medicine) > 0:
            top_medicine_name = str(top_medicine.iloc[0]['medicine'])
            top_medicine_count = int(top_medicine.iloc[0]['count'])
        else:
            top_medicine_name = 'N/A'
            top_medicine_count = 0
    except:
        top_medicine_name = 'N/A'
        top_medicine_count = 0
    
    try:
        seasonal = pd.read_sql_query("""
            SELECT season, COUNT(*) as count 
            FROM prescriptions 
            GROUP BY season
        """, conn)
        seasonal_data = []
        for _, row in seasonal.iterrows():
            seasonal_data.append({
                'season': str(row['season']),
                'count': int(row['count'])
            })
    except:
        seasonal_data = []
    
    conn.close()
    
    return {
        'total': total,
        'top_disease': top_disease_name,
        'top_disease_count': top_disease_count,
        'top_medicine': top_medicine_name,
        'top_medicine_count': top_medicine_count,
        'seasonal': seasonal_data
    }

def delete_record(record_id):
    """Delete a record by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prescriptions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    print(f"[✓] Record {record_id} deleted")

# Initialize database when module loads
init_db()