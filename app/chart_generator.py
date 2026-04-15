import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
from database import get_all_records

# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def get_chart_as_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def generate_disease_chart():
    """Generate real-time disease frequency chart"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    disease_counts = df['disease'].value_counts()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    bars = ax.bar(disease_counts.index, disease_counts.values, color=colors, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, disease_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', fontweight='bold', fontsize=11)
    
    ax.set_title('Disease Frequency (Real-Time Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=12)
    ax.set_ylabel('Number of Prescriptions', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_age_chart():
    """Generate real-time age group distribution"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    def age_group(age):
        if age <= 18: return 'Children (0-18)'
        elif age <= 60: return 'Adults (19-60)'
        else: return 'Elderly (60+)'
    
    df['age_group'] = df['age'].apply(age_group)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    age_counts = df['age_group'].value_counts()
    axes[0].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', 
                colors=['#1f77b4', '#ff7f0e', '#2ca02c'], startangle=90)
    axes[0].set_title('Age Group Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart by disease
    pivot = df.groupby(['disease', 'age_group']).size().unstack(fill_value=0)
    pivot.plot(kind='bar', ax=axes[1], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    axes[1].set_title('Disease by Age Group', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Disease', fontsize=10)
    axes[1].set_ylabel('Count', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Age Group')
    
    plt.suptitle('Age-Based Analysis (Real-Time Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_seasonal_chart():
    """Generate real-time seasonal trends chart"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get season counts
    seasonal_counts = df['season'].value_counts()
    season_order = ['Winter', 'Spring', 'Monsoon', 'Autumn']
    season_colors = {'Winter': '#4e79a7', 'Spring': '#f28e2b', 'Monsoon': '#59a14f', 'Autumn': '#e15759'}
    colors = [season_colors.get(s, '#1f77b4') for s in seasonal_counts.index]
    
    bars = ax.bar(seasonal_counts.index, seasonal_counts.values, color=colors, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, seasonal_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(val), ha='center', fontweight='bold', fontsize=11)
    
    ax.set_title('Seasonal Disease Distribution (Real-Time Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Number of Prescriptions', fontsize=12)
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_gender_chart():
    """Generate real-time gender distribution chart"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_disease = pd.crosstab(df['disease'], df['gender'])
    gender_disease.plot(kind='bar', ax=ax, color=['#e41a1c', '#377eb8'], edgecolor='black', linewidth=1)
    
    ax.set_title('Disease Distribution by Gender (Real-Time Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Gender', labels=['Female', 'Male'])
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=3)
    
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_medicine_chart():
    """Generate real-time top medicines chart"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    med_counts = df['medicine'].value_counts().head(10)
    bars = ax.barh(med_counts.index[::-1], med_counts.values[::-1], 
                   color='#2ca02c', edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, med_counts.values[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                str(val), va='center', fontweight='bold', fontsize=11)
    
    ax.set_title('Top 10 Most Prescribed Medicines (Real-Time Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Prescriptions', fontsize=12)
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_feature_importance():
    """Generate feature importance chart (static - from model)"""
    import joblib
    import os
    
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS = os.path.join(BASE, 'models')
    model_path = os.path.join(MODELS, 'random_forest.pkl')
    
    if not os.path.exists(model_path):
        return None
    
    model = joblib.load(model_path)
    features = ['age', 'gender', 'medicine', 'dosage', 'frequency', 'month']
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(importances.index, importances.values, color='#9467bd', edgecolor='black')
    
    for bar, val in zip(bars, importances.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontweight='bold')
    
    ax.set_title('Feature Importance (Random Forest Model)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

def generate_seasonal_heatmap():
    """Generate real-time seasonal disease heatmap from database"""
    df = get_all_records()
    
    if df.empty:
        return None
    
    # Create season column
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Monsoon'
        else:
            return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    # Create cross-tabulation (Season vs Disease)
    seasonal_disease = pd.crosstab(df['season'], df['disease'])
    
    # Define season order
    season_order = ['Winter', 'Spring', 'Monsoon', 'Autumn']
    seasonal_disease = seasonal_disease.reindex(season_order)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(seasonal_disease, annot=True, fmt='d', cmap='YlOrRd', 
                ax=ax, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Number of Prescriptions', 'shrink': 0.8})
    
    ax.set_title('Seasonal Disease Distribution (Real-Time Data)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Disease', fontsize=12, fontweight='bold')
    ax.set_ylabel('Season', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return get_chart_as_base64(fig)

print("[OK] Real-time chart generator loaded")