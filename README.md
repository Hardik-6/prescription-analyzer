# Common Disease Analysis Using Prescription Data
## Capstone Project — Sahil Parte (61) & Hardik Patil (68)
### BE Computer Engineering, 2025

---

## Project Overview

A machine learning system that analyses prescription data to predict
common diseases. The system compares 4 ML algorithms and deploys
the best model as a Flask web application.

---

## Project Structure

```
capstone/
├── data/
│   └── prescriptions.csv       ← Dataset (100 records, 8 columns)
├── models/
│   ├── random_forest.pkl       ← Trained best model
│   ├── le_gender.pkl           ← Gender label encoder
│   ├── le_medicine.pkl         ← Medicine label encoder
│   └── le_disease.pkl          ← Disease label encoder
├── plots/
│   ├── chart1_disease_frequency.png
│   ├── chart2_age_distribution.png
│   ├── chart3_seasonal_trends.png
│   ├── chart4_medicine_demand.png
│   ├── chart5_model_comparison.png
│   ├── chart6_confusion_matrices.png
│   └── chart7_feature_importance.png
├── app/
│   ├── app.py                  ← Flask web application
│   └── templates/
│       └── index.html          ← Web UI
├── analysis.py                 ← Full ML pipeline script
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Step 1 — Install Python (3.9 or above)
Download from https://python.org

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the analysis (generates all charts + trains models)
```bash
python analysis.py
```

### Step 4 — Run the web app
```bash
cd app
python app.py
```
Then open your browser at: http://localhost:5000

---

## Dataset

- 100 real-world prescription records
- Collected from local clinics and medical stores
- 7 diseases: Fever, Diabetes, Hypertension, Cold & Cough,
              Gastric, Infection, Allergy
- Features: age, gender, medicine, dosage, frequency, month

---

## ML Models & Results

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Naïve Bayes   | 80.0%    | 90.3%     | 80.0%  | 78.6%    |
| SVM           | 40.0%    | 21.2%     | 40.0%  | 25.7%    |
| KNN           | 60.0%    | 76.4%     | 60.0%  | 58.2%    |
| Random Forest | 95.0%    | 96.3%     | 95.0%  | 94.9%    |

**Best Model: Random Forest — 95% Accuracy**

---

## Key Findings

1. Fever and Cold & Cough peak during monsoon months (July–September)
2. Elderly patients (60+) show higher prevalence of Diabetes and Hypertension
3. Paracetamol and Metformin are the most prescribed medicines
4. Random Forest outperforms all other models significantly
5. Medicine name and age are the most important features for prediction

---

## Technologies Used

- Python 3.x
- pandas — data manipulation
- scikit-learn — machine learning
- matplotlib & seaborn — data visualization
- Flask — web application
- joblib — model persistence
