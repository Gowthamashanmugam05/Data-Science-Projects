"""
Supermart Grocery Sales - Retail Analytics & Sales Prediction
===============================================================
A comprehensive data analytics and machine learning project to analyze
grocery sales data, identify trends, and predict sales using Linear Regression.

Author: Data Analyst & Data Scientist
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("SUPERMART GROCERY SALES - RETAIL ANALYTICS & SALES PREDICTION")
print("=" * 80)

# ==============================================================================
# STEP 1: ENVIRONMENT SETUP & STEP 2: LOAD DATASET
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 1 & 2: ENVIRONMENT SETUP & DATA LOADING")
print("=" * 80)

# Load dataset
try:
    import os
    # Try multiple path configurations
    possible_paths = [
        '../data/supermart_grocery_sales.csv',
        'data/supermart_grocery_sales.csv',
        './data/supermart_grocery_sales.csv'
    ]
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    if df is None:
        raise FileNotFoundError("Dataset not found in any expected location")
    print("\n✓ Dataset loaded successfully!")
except FileNotFoundError:
    print("\n✗ ERROR: Dataset file not found!")
    print("Please ensure 'supermart_grocery_sales.csv' exists in the data/ folder")
    exit(1)

# Display basic information
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Names & Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Records ---")
print(f"Total duplicates: {df.duplicated().sum()}")

# Base and output directories (ensure output exists)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# STEP 3: DATA CLEANING & PREPROCESSING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA CLEANING & PREPROCESSING")
print("=" * 80)

# 3.1 Handle Missing & Duplicate Data
print("\n--- Removing Missing Values & Duplicates ---")
initial_rows = len(df)
df = df.dropna()
df = df.drop_duplicates()
final_rows = len(df)
print(f"Rows removed: {initial_rows - final_rows}")
print(f"Dataset shape after cleaning: {df.shape}")

# 3.2 Date Handling
print("\n--- Date Feature Engineering ---")
# Handle multiple date formats
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)

# Extract date features
df['month_no'] = df['Order Date'].dt.month
df['Month'] = df['Order Date'].dt.month_name()
df['year'] = df['Order Date'].dt.year

print("Date columns created: month_no, Month, year")
print("\nSample of date-engineered data:")
print(df[['Order Date', 'month_no', 'Month', 'year']].head())

# ==============================================================================
# STEP 4: ENCODING CATEGORICAL VARIABLES
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 4: ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# Initialize label encoders
label_encoders = {}
categorical_cols = ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']

print("\n--- Applying Label Encoding ---")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"✓ {col} encoded")

print("\n--- Dataset Head After Encoding ---")
print(df.head())

# ==============================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Create figure for EDA plots
plt.figure(figsize=(16, 12))

# 5.1 Category-Level Analysis
print("\n--- 5.1: CATEGORY-LEVEL ANALYSIS ---")

# Reload original data temporarily for visualization purposes
# Reload original data temporarily for visualization purposes
possible_paths = [
    '../data/supermart_grocery_sales.csv',
    'data/supermart_grocery_sales.csv',
    './data/supermart_grocery_sales.csv'
]
df_original = None
for path in possible_paths:
    try:
        df_original = pd.read_csv(path)
        break
    except FileNotFoundError:
        continue
if df_original is None:
    print("Warning: Could not reload original data")
    df_original = df.copy()
else:
    # Parse dates robustly for original dataframe used in visualization
    df_original['Order Date'] = pd.to_datetime(
        df_original['Order Date'], dayfirst=True, errors='coerce', infer_datetime_format=True
    )
    # Drop rows where parsing failed
    if df_original['Order Date'].isnull().any():
        df_original = df_original.dropna(subset=['Order Date'])

category_sales = df_original.groupby('Category')['Sales'].sum().sort_values(ascending=False)

plt.subplot(3, 3, 1)
category_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Total Sales by Category', fontsize=12, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

print("\nTotal Sales by Category:")
print(category_sales)
print(f"\n💡 Insight: {category_sales.index[0]} leads with ${category_sales.values[0]:,.2f} in sales.")

# 5.2 Time-Based Analysis
print("\n--- 5.2: TIME-BASED ANALYSIS ---")

# Monthly Sales
df_original['month_no'] = df_original['Order Date'].dt.month
monthly_sales = df_original.groupby('month_no')['Sales'].sum().sort_index()

plt.subplot(3, 3, 2)
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='green')
plt.title('Monthly Sales Trend', fontsize=12, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Sales ($)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))

print("\nMonthly Sales:")
print(monthly_sales)

# Yearly Sales
df_original['year'] = df_original['Order Date'].dt.year
yearly_sales = df_original.groupby('year')['Sales'].sum()

plt.subplot(3, 3, 3)
colors = plt.cm.Set3(range(len(yearly_sales)))
plt.pie(yearly_sales.values, labels=yearly_sales.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Sales Share by Year', fontsize=12, fontweight='bold')

print(f"\nYearly Sales:")
print(yearly_sales)
print(f"\n💡 Insight: Peak sales in {yearly_sales.idxmax()} with ${yearly_sales.max():,.2f}")

# 5.3 City-Level Analysis
print("\n--- 5.3: CITY-LEVEL ANALYSIS ---")

city_sales = df_original.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)

plt.subplot(3, 3, 4)
city_sales.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Top 5 Cities by Total Sales', fontsize=12, fontweight='bold')
plt.xlabel('City')
plt.ylabel('Total Sales ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

print("\nTop 5 Cities by Sales:")
print(city_sales)
print(f"\n💡 Insight: {city_sales.index[0]} is the top city with ${city_sales.values[0]:,.2f} in sales.")

# 5.4 Correlation Analysis
print("\n--- 5.4: CORRELATION ANALYSIS ---")

# Select numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.subplot(3, 3, 5)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            cbar_kws={'label': 'Correlation'}, square=True)
plt.title('Correlation Heatmap', fontsize=12, fontweight='bold')
plt.tight_layout()

# Find strong correlations with Sales
sales_corr = correlation_matrix['Sales'].sort_values(ascending=False)
print("\nCorrelation with Sales:")
print(sales_corr)
print(f"\n💡 Insight: Profit has strongest correlation with Sales ({sales_corr['Profit']:.3f})")

# Additional visualizations
# Discount vs Sales
plt.subplot(3, 3, 6)
plt.scatter(df_original['Discount'], df_original['Sales'], alpha=0.5, color='purple')
plt.title('Discount vs Sales', fontsize=12, fontweight='bold')
plt.xlabel('Discount')
plt.ylabel('Sales ($)')
plt.grid(True, alpha=0.3)

# Profit Distribution
plt.subplot(3, 3, 7)
df_original['Profit'].hist(bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title('Profit Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Profit ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Sales Distribution
plt.subplot(3, 3, 8)
df_original['Sales'].hist(bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Sales Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Region-wise Sales
plt.subplot(3, 3, 9)
region_sales = df_original.groupby('Region')['Sales'].sum().sort_values(ascending=False)
region_sales.plot(kind='barh', color='teal', edgecolor='black')
plt.title('Sales by Region', fontsize=12, fontweight='bold')
plt.xlabel('Total Sales ($)')
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
eda_path = os.path.join(OUTPUT_DIR, '01_EDA_Analysis.png')
plt.savefig(eda_path, dpi=300, bbox_inches='tight')
print(f"\n✓ EDA plots saved to '{eda_path}'")
plt.show()

# ==============================================================================
# STEP 6: FEATURE SELECTION FOR ML
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 6: FEATURE SELECTION FOR ML")
print("=" * 80)

# Define features and target
feature_columns = ['Category', 'Sub Category', 'City', 'Region', 'State', 
                   'month_no', 'Discount', 'Profit']
target_column = 'Sales'

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"\nFeatures (X): {feature_columns}")
print(f"Target (y): {target_column}")
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Columns to drop
drop_columns = ['Order ID', 'Customer Name', 'Order Date', 'Month']
print(f"\nDropped columns: {drop_columns}")

# ==============================================================================
# STEP 7: TRAIN-TEST SPLIT & SCALING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 7: TRAIN-TEST SPLIT & SCALING")
print("=" * 80)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples (80%)")
print(f"Testing set size: {X_test.shape[0]} samples (20%)")

# Apply StandardScaler to features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ StandardScaler applied to features")
print(f"Training features shape (scaled): {X_train_scaled.shape}")
print(f"Testing features shape (scaled): {X_test_scaled.shape}")

# ==============================================================================
# STEP 8: MODEL TRAINING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 8: MODEL TRAINING")
print("=" * 80)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("\n✓ Linear Regression model trained successfully!")
print(f"\nModel Coefficients (Feature Importance):")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', ascending=False)
print(feature_importance)
print(f"\nIntercept: {lr_model.intercept_:.4f}")

# Predict on test data
y_pred = lr_model.predict(X_test_scaled)

print(f"\n✓ Sales predictions generated for {len(y_pred)} test samples")

# ==============================================================================
# STEP 9: MODEL EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 9: MODEL EVALUATION")
print("=" * 80)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "─" * 80)
print("MODEL PERFORMANCE METRICS")
print("─" * 80)
print(f"Mean Squared Error (MSE):  {mse:,.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score:                  {r2:.4f} ({r2*100:.2f}%)")
print("─" * 80)

# Interpretation
if r2 >= 0.7:
    r2_interpretation = "EXCELLENT - Model explains most variance"
elif r2 >= 0.5:
    r2_interpretation = "GOOD - Model explains substantial variance"
else:
    r2_interpretation = "FAIR - Model explains limited variance"

print(f"R² Interpretation: {r2_interpretation}")
print("─" * 80)

# ==============================================================================
# STEP 10: MODEL VISUALIZATION
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 10: MODEL VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted Sales
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolor='black')
# Ideal prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Sales ($)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Sales ($)', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted Sales', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolor='black')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Sales ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Distribution
ax3 = axes[1, 0]
ax3.hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature Importance
ax4 = axes[1, 1]
top_features = feature_importance.head(8)
ax4.barh(top_features['Feature'], top_features['Coefficient'], color='purple', edgecolor='black')
ax4.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax4.set_title('Top 8 Feature Importance', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
model_vis_path = os.path.join(OUTPUT_DIR, '02_Model_Visualization.png')
plt.savefig(model_vis_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Model visualization plots saved to '{model_vis_path}'")
plt.show()

# ==============================================================================
# STEP 11: BUSINESS INSIGHTS
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 11: BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

print("\n" + "═" * 80)
print("1. BEST PERFORMING CATEGORIES")
print("═" * 80)
top_categories = df_original.groupby('Category')['Sales'].sum().sort_values(ascending=False)
for i, (cat, sales) in enumerate(top_categories.items(), 1):
    print(f"   {i}. {cat:20s}: ${sales:>12,.2f}")

print("\n" + "═" * 80)
print("2. CITIES WITH HIGHEST DEMAND")
print("═" * 80)
top_cities = df_original.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)
for i, (city, sales) in enumerate(top_cities.items(), 1):
    print(f"   {i}. {city:20s}: ${sales:>12,.2f}")

print("\n" + "═" * 80)
print("3. MONTHLY & YEARLY GROWTH PATTERN")
print("═" * 80)
print("\nMonthly Average Sales:")
monthly_avg = df_original.groupby('month_no')['Sales'].mean()
for month, avg_sales in monthly_avg.items():
    print(f"   Month {month:2d}: ${avg_sales:>10,.2f}")

yearly_avg = df_original.groupby('year')['Sales'].mean()
print("\nYearly Average Sales:")
for year, avg_sales in yearly_avg.items():
    print(f"   {year}: ${avg_sales:>10,.2f}")

print("\n" + "═" * 80)
print("4. IMPACT OF DISCOUNT & PROFIT ON SALES")
print("═" * 80)
discount_corr = df_original['Discount'].corr(df_original['Sales'])
profit_corr = df_original['Profit'].corr(df_original['Sales'])
print(f"   Discount correlation with Sales: {discount_corr:>8.4f}")
print(f"   Profit correlation with Sales:   {profit_corr:>8.4f}")

if profit_corr > 0.7:
    print(f"\n   💡 INSIGHT: Profit is STRONGLY correlated with Sales")
    print(f"      → Focus on high-margin products for revenue growth")
if abs(discount_corr) < 0.3:
    print(f"\n   💡 INSIGHT: Discounts have WEAK correlation with Sales")
    print(f"      → Strategic discounting has minimal impact on volume")
else:
    print(f"\n   💡 INSIGHT: Discounts show MODERATE correlation with Sales")
    print(f"      → Consider balanced discount strategy")

print("\n" + "═" * 80)
print("5. MODEL PERFORMANCE SUMMARY")
print("═" * 80)
print(f"   R² Score:     {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"   RMSE:         ${rmse:,.2f}")
print(f"   MSE:          {mse:,.4f}")

print("\n   📊 Model Assessment:")
if r2 >= 0.7:
    print("   ✓ Model has STRONG predictive power")
    print("   ✓ Suitable for real-world sales forecasting")
elif r2 >= 0.5:
    print("   ✓ Model has MODERATE predictive power")
    print("   ⚠ Consider feature engineering for improvement")
else:
    print("   ⚠ Model has LIMITED predictive power")
    print("   ⚠ Additional features or model refinement recommended")

print("\n" + "═" * 80)
print("6. STRATEGIC RECOMMENDATIONS")
print("═" * 80)
print("   1. Focus marketing budget on top-performing categories")
print("   2. Expand distribution in high-demand cities")
print("   3. Implement seasonal inventory management based on monthly trends")
print("   4. Prioritize profit margins over aggressive discounting")
print("   5. Develop predictive models for demand forecasting")
print("   6. Monitor competitor activity in top-performing regions")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "═" * 80)
print("PROJECT SUMMARY")
print("═" * 80)
print(f"\n✓ Dataset Size: {df_original.shape[0]:,} transactions")
print(f"✓ Total Sales: ${df_original['Sales'].sum():,.2f}")
print(f"✓ Average Transaction: ${df_original['Sales'].mean():,.2f}")
print(f"✓ Features Used: {len(feature_columns)}")
print(f"✓ Training Samples: {X_train.shape[0]:,}")
print(f"✓ Testing Samples: {X_test.shape[0]:,}")
print(f"✓ Model Type: Linear Regression")
print(f"✓ Model R² Score: {r2:.4f}")

print("\n" + "═" * 80)
print("✅ ANALYSIS COMPLETE - ALL STEPS EXECUTED SUCCESSFULLY!")
print("═" * 80)
print("\nGenerated Outputs:")
print("  • EDA Analysis plots: output/01_EDA_Analysis.png")
print("  • Model Visualization: output/02_Model_Visualization.png")
print("  • Console insights and metrics (above)")
print("\n" + "═" * 80)
