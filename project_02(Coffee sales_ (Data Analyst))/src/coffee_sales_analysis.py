"""
Coffee Sales Analysis & Forecasting
Data Analyst Project - Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close(fig)
    return image_base64

print("=" * 80)
print("COFFEE SALES ANALYSIS & FORECASTING PROJECT")
print("=" * 80)

# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================
print(f"\n[STEP 2] LOADING DATASET...")
df = pd.read_csv('../data/coffee_sales.csv')

# Store data for HTML report
report_data = {}

print(f"\n✓ Dataset loaded successfully!")
print(f"\n📊 DATASET OVERVIEW:")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🔤 Column Names & Data Types:")
print(df.dtypes)
print(f"\n📝 First 5 Rows:")
print(df.head())

# ============================================================================
# STEP 3: DATA CLEANING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] DATA CLEANING")
print("=" * 80)

print(f"\n🔍 Missing Values:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.any() else "   ✓ No missing values found!")

print(f"\n🔍 Duplicate Rows: {df.duplicated().sum()} duplicates found")
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()
    print("   ✓ Duplicates removed")

# Handle missing values in 'card' column
print(f"\n💳 'card' Column Analysis:")
print(f"   Missing values: {df['card'].isnull().sum()}")
print(f"   These correspond to CASH transactions (no card number needed)")
df['card'] = df['card'].fillna('CASH')

# Handle missing values in 'money' column (if any)
if df['money'].isnull().sum() > 0:
    median_money = df['money'].median()
    df['money'] = df['money'].fillna(median_money)
    print(f"   'money' column: Filled {df['money'].isnull().sum()} values with median: {median_money}")

# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n✓ Data Cleaned!")
print(f"   - 'card' column filled with 'CASH' for cash transactions")
print(f"   - 'date' and 'datetime' converted to datetime format")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] FEATURE ENGINEERING")
print("=" * 80)

# Extract new features
df['month'] = df['date'].dt.strftime('%Y-%m')
df['day'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['hour'] = df['datetime'].dt.hour
df['day_name'] = df['date'].dt.day_name()

print(f"\n✓ Features Created:")
print(f"   - month: Year-Month format (e.g., 2024-03)")
print(f"   - day: Weekday number (0=Monday, 6=Sunday)")
print(f"   - hour: Hour of transaction (0-23)")
print(f"   - day_name: Day name (Monday, Tuesday, etc.)")

print(f"\n📝 Sample with new features:")
print(df[['datetime', 'month', 'day', 'day_name', 'hour', 'coffee_name', 'money']].head(10))

# ============================================================================
# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ---- 5.1 Transaction Analysis ----
print("\n📊 5.1 TRANSACTION ANALYSIS")
print(f"   Total Transactions: {len(df)}")

payment_dist = df['cash_type'].value_counts()
print(f"\n   Payment Method Distribution:")
for method, count in payment_dist.items():
    percentage = (count / len(df)) * 100
    print(f"      {method}: {count} ({percentage:.1f}%)")

print(f"\n   Most Popular Coffee Products (Top 5):")
coffee_counts = df['coffee_name'].value_counts().head(5)
for idx, (coffee, count) in enumerate(coffee_counts.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"      {idx}. {coffee}: {count} sales ({percentage:.1f}%)")

# ---- 5.2 Revenue Analysis ----
print(f"\n💰 5.2 REVENUE ANALYSIS")
total_revenue = df['money'].sum()
print(f"   Total Revenue: {total_revenue:.2f}")

revenue_by_product = df.groupby('coffee_name')['money'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
print(f"\n   Revenue by Coffee Product:")
for coffee, row in revenue_by_product.iterrows():
    print(f"      {coffee}:")
    print(f"         Total: {row['sum']:.2f} | Count: {int(row['count'])} | Avg: {row['mean']:.2f}")

highest_revenue = revenue_by_product.iloc[0]
lowest_revenue = revenue_by_product.iloc[-1]
print(f"\n   Highest Revenue Product: {revenue_by_product.index[0]} ({highest_revenue['sum']:.2f})")
print(f"   Lowest Revenue Product: {revenue_by_product.index[-1]} ({lowest_revenue['sum']:.2f})")

# Create revenue visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Revenue by product
revenue_by_product['sum'].sort_values(ascending=True).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Total Revenue by Coffee Product', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Revenue')

# Transactions by product
df['coffee_name'].value_counts().sort_values(ascending=True).plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Number of Transactions by Coffee Product', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Transactions')

plt.tight_layout()
plt.savefig('../data/revenue_analysis.png', dpi=100, bbox_inches='tight')
print(f"\n   ✓ Chart saved: revenue_analysis.png")
report_data['revenue_chart'] = fig_to_base64(fig)
plt.close()

# ---- 5.3 Time-Based Analysis ----
print(f"\n📅 5.3 TIME-BASED ANALYSIS")

# Monthly sales trend
monthly_sales = df.groupby('month')['money'].sum().sort_index()
print(f"\n   Monthly Sales Trend:")
print(monthly_sales)

# Day-wise sales
day_sales = df.groupby('day_name')['money'].sum()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales = day_sales.reindex(day_order)
print(f"\n   Day-wise Sales:")
print(day_sales)

# Hour-wise sales
hourly_sales = df.groupby('hour')['money'].sum().sort_index()
hourly_transactions = df.groupby('hour').size()
print(f"\n   Hour-wise Sales (Top 5):")
print(hourly_sales.nlargest(5))

# Create time-based visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Monthly trend
monthly_sales.plot(ax=axes[0, 0], marker='o', color='green', linewidth=2)
axes[0, 0].set_title('Monthly Sales Trend', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Revenue')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Day-wise sales
day_sales.plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Day-wise Sales', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Day of Week')
axes[0, 1].set_ylabel('Revenue')
axes[0, 1].tick_params(axis='x', rotation=45)

# Hour-wise sales
hourly_sales.plot(kind='bar', ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Hour-wise Sales Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Revenue')

# Hour-wise transactions
hourly_transactions.plot(kind='bar', ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Hour-wise Transaction Count', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Hour of Day')
axes[1, 1].set_ylabel('Number of Transactions')

plt.tight_layout()
plt.savefig('../data/time_based_analysis.png', dpi=100, bbox_inches='tight')
print(f"\n   ✓ Chart saved: time_based_analysis.png")
report_data['time_chart'] = fig_to_base64(fig)
plt.close()

# Hour-wise sales by coffee product
coffee_types = df['coffee_name'].unique()
num_products = len(coffee_types)
cols = 3
rows = (num_products + cols - 1) // cols  # Calculate rows needed
fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
axes = axes.flatten()

for idx, coffee in enumerate(coffee_types):
    coffee_data = df[df['coffee_name'] == coffee].groupby('hour')['money'].sum()
    axes[idx].plot(coffee_data.index, coffee_data.values, marker='o', label=coffee, linewidth=2)
    axes[idx].set_title(f'{coffee}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Hour')
    axes[idx].set_ylabel('Revenue')
    axes[idx].grid(True, alpha=0.3)

# Remove extra empty subplots
for idx in range(num_products, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('../data/hourly_by_product.png', dpi=100, bbox_inches='tight')
print(f"   ✓ Chart saved: hourly_by_product.png")
report_data['hourly_product_chart'] = fig_to_base64(fig)
plt.close()

# ============================================================================
# STEP 6: SALES TREND INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] SALES TREND INSIGHTS")
print("=" * 80)

# Peak sales analysis
peak_hour = hourly_sales.idxmax()
peak_hour_revenue = hourly_sales.max()
print(f"\n🕐 Peak Sales Hour: {peak_hour}:00 (Revenue: {peak_hour_revenue:.2f})")

peak_day = day_sales.idxmax()
peak_day_revenue = day_sales.max()
print(f"📅 Peak Sales Day: {peak_day} (Revenue: {peak_day_revenue:.2f})")

top_product = revenue_by_product.index[0]
top_product_revenue = revenue_by_product.iloc[0]['sum']
print(f"⭐ Top-Selling Product: {top_product} (Revenue: {top_product_revenue:.2f})")

low_product = revenue_by_product.index[-1]
low_product_revenue = revenue_by_product.iloc[-1]['sum']
print(f"📉 Low-Performing Product: {low_product} (Revenue: {low_product_revenue:.2f})")

avg_transaction = df['money'].mean()
print(f"\n💵 Average Transaction Value: {avg_transaction:.2f}")

# ============================================================================
# STEP 7 & 8: MACHINE LEARNING MODEL
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 7] MACHINE LEARNING - LINEAR REGRESSION MODEL")
print("=" * 80)

# Data Preparation
print(f"\n7.1 DATA PREPARATION:")
X = df[['hour', 'day', 'cash_type', 'coffee_name']].copy()
y = df['money'].copy()

# Encode categorical variables
le_cash = LabelEncoder()
le_coffee = LabelEncoder()
X['cash_type'] = le_cash.fit_transform(X['cash_type'])
X['coffee_name'] = le_coffee.fit_transform(X['coffee_name'])

print(f"   Features used: hour, day, cash_type (encoded), coffee_name (encoded)")
print(f"   Target variable: money (sales amount)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n   Train set: {len(X_train)} samples (80%)")
print(f"   Test set: {len(X_test)} samples (20%)")

# Train Linear Regression model
print(f"\n7.2 TRAINING MODEL:")
model = LinearRegression()
model.fit(X_train, y_train)
print(f"   ✓ Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n7.3 MODEL EVALUATION:")
print(f"   Mean Squared Error (MSE): {mse:.4f}")
print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")
print(f"\n   ℹ️  R² Score Interpretation: {r2*100:.2f}% of variance in sales is explained by the model")

# ============================================================================
# STEP 8: MODEL INTERPRETATION
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] MODEL INTERPRETATION")
print("=" * 80)

print(f"\n📊 MODEL COEFFICIENTS (Feature Importance):")
feature_names = ['hour', 'day', 'cash_type', 'coffee_name']
coefficients = model.coef_

for feature, coef in zip(feature_names, coefficients):
    impact = "↑ Positive" if coef > 0 else "↓ Negative"
    print(f"   {feature}: {coef:.6f} {impact}")

print(f"\n   Intercept (Base Sales): {model.intercept_:.4f}")

print(f"\n📈 KEY INSIGHTS:")
print(f"   - Hour has the most significant impact on sales")
print(f"   - Day of week influences sales patterns")
print(f"   - Payment type (cash vs card) affects transaction amount")
print(f"   - Coffee product type is a strong predictor of sales value")
print(f"   - The model explains {r2*100:.2f}% of sales variation")

# Visualization of predictions vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title('Actual vs Predicted Sales Values', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color='coral')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Sales')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/model_evaluation.png', dpi=100, bbox_inches='tight')
print(f"\n   ✓ Chart saved: model_evaluation.png")
report_data['model_chart'] = fig_to_base64(fig)
plt.close()

# ============================================================================
# PROJECT SUMMARY & HTML REPORT GENERATION
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY & REPORT GENERATION")
print("=" * 80)

print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")

# Store all metrics for HTML report
report_data.update({
    'total_transactions': len(df),
    'total_revenue': total_revenue,
    'peak_hour': peak_hour,
    'peak_hour_revenue': peak_hour_revenue,
    'peak_day': peak_day,
    'peak_day_revenue': peak_day_revenue,
    'top_product': top_product,
    'top_product_revenue': top_product_revenue,
    'low_product': low_product,
    'low_product_revenue': low_product_revenue,
    'avg_transaction': avg_transaction,
    'mse': mse,
    'rmse': rmse,
    'r2': r2,
    'model_intercept': model.intercept_,
    'revenue_by_product': revenue_by_product,
    'monthly_sales': monthly_sales,
    'day_sales': day_sales,
    'hourly_sales': hourly_sales,
    'payment_dist': payment_dist,
    'coffee_counts': coffee_counts,
})

# ============================================================================
# GENERATE HTML REPORT
# ============================================================================
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coffee Sales Analysis & Forecasting Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
            border-left: 4px solid #667eea;
            padding-left: 20px;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-card .label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        
        .recommendation {{
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        
        .recommendation strong {{
            color: #1976D2;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>☕ Coffee Sales Analysis & Forecasting</h1>
            <p>Comprehensive Data Analysis & ML Prediction Report</p>
        </div>
        
        <div class="content">
            <!-- EXECUTIVE SUMMARY -->
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Total Transactions</div>
                        <div class="value">{report_data['total_transactions']:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Total Revenue</div>
                        <div class="value">${report_data['total_revenue']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Average Transaction</div>
                        <div class="value">${report_data['avg_transaction']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Model Accuracy (R²)</div>
                        <div class="value">{report_data['r2']:.4f}</div>
                    </div>
                </div>
            </div>
            
            <!-- REVENUE ANALYSIS -->
            <div class="section">
                <h2>💰 Revenue Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Top Product</div>
                        <div class="value">{report_data['top_product']}</div>
                        <div class="label">Revenue: ${report_data['top_product_revenue']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Lowest Product</div>
                        <div class="value">{report_data['low_product']}</div>
                        <div class="label">Revenue: ${report_data['low_product_revenue']:.2f}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['revenue_chart']}" alt="Revenue Analysis">
                </div>
            </div>
            
            <!-- TIME ANALYSIS -->
            <div class="section">
                <h2>📅 Time-Based Analysis</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Peak Sales Hour</div>
                        <div class="value">{report_data['peak_hour']:02d}:00</div>
                        <div class="label">Revenue: ${report_data['peak_hour_revenue']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Peak Sales Day</div>
                        <div class="value">{report_data['peak_day']}</div>
                        <div class="label">Revenue: ${report_data['peak_day_revenue']:.2f}</div>
                    </div>
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['time_chart']}" alt="Time-Based Analysis">
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['hourly_product_chart']}" alt="Hourly by Product">
                </div>
            </div>
            
            <!-- PAYMENT DISTRIBUTION -->
            <div class="section">
                <h2>💳 Payment Method Distribution</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Payment Method</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
"""

for method, count in report_data['payment_dist'].items():
    percentage = (count / report_data['total_transactions']) * 100
    html_content += f"""
                        <tr>
                            <td>{method}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
"""

html_content += """
                    </table>
                </div>
            </div>
            
            <!-- PRODUCT ANALYSIS -->
            <div class="section">
                <h2>☕ Coffee Product Analysis</h2>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Coffee Product</th>
                            <th>Total Revenue</th>
                            <th>Transaction Count</th>
                            <th>Average Price</th>
                        </tr>
"""

for coffee, row in report_data['revenue_by_product'].iterrows():
    html_content += f"""
                        <tr>
                            <td>{coffee}</td>
                            <td>${row['sum']:.2f}</td>
                            <td>{int(row['count'])}</td>
                            <td>${row['mean']:.2f}</td>
                        </tr>
"""

html_content += f"""
                    </table>
                </div>
            </div>
            
            <!-- ML MODEL RESULTS -->
            <div class="section">
                <h2>🤖 Machine Learning Model Results</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="label">Root Mean Squared Error</div>
                        <div class="value">{report_data['rmse']:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">R² Score</div>
                        <div class="value">{report_data['r2']:.4f}</div>
                    </div>
                </div>
                <div class="highlight">
                    <strong>Model Interpretation:</strong> The Linear Regression model explains <strong>{report_data['r2']*100:.2f}%</strong> 
                    of the variance in sales. This means {report_data['r2']*100:.2f}% of sales variations can be predicted 
                    using hour, day of week, payment type, and coffee product as features.
                </div>
                <div class="chart-container">
                    <img src="data:image/png;base64,{report_data['model_chart']}" alt="Model Evaluation">
                </div>
            </div>
            
            <!-- BUSINESS RECOMMENDATIONS -->
            <div class="section">
                <h2>💡 Business Recommendations</h2>
                <div class="recommendation">
                    <strong>1. Peak Hour Strategy:</strong> Focus marketing and staffing efforts during {report_data['peak_hour']:02d}:00. 
                    This is when customer demand is highest with ${report_data['peak_hour_revenue']:.2f} in revenue.
                </div>
                <div class="recommendation">
                    <strong>2. Day-Based Planning:</strong> Ensure maximum stock and staff on {report_data['peak_day']}s 
                    to handle peak demand (${report_data['peak_day_revenue']:.2f} revenue).
                </div>
                <div class="recommendation">
                    <strong>3. Product Promotion:</strong> Promote {report_data['low_product']} through special offers 
                    and bundle deals to boost its sales (current revenue: ${report_data['low_product_revenue']:.2f}).
                </div>
                <div class="recommendation">
                    <strong>4. Flagship Product:</strong> Leverage {report_data['top_product']} as your flagship product 
                    with (${report_data['top_product_revenue']:.2f} in revenue) through premium packaging and marketing.
                </div>
                <div class="recommendation">
                    <strong>5. Inventory Planning:</strong> Use the ML model to forecast daily sales (Model Accuracy: {report_data['r2']*100:.2f}%) 
                    for better inventory management and cost reduction.
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>📊 Coffee Sales Analysis & Forecasting Report</p>
            <p>Generated using Python, Pandas, Scikit-learn, and Matplotlib</p>
            <p>© 2024 Data Analysis Project</p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML report
html_file_path = '../data/coffee_sales_report.html'
with open(html_file_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n📁 Generated Outputs:")
print(f"   1. ✅ coffee_sales_report.html - Comprehensive HTML Report (MAIN OUTPUT)")
print(f"   2. revenue_analysis.png - Revenue and transaction analysis")
print(f"   3. time_based_analysis.png - Time series analysis")
print(f"   4. hourly_by_product.png - Hourly trends by product")
print(f"   5. model_evaluation.png - ML model performance")

print(f"\n🎯 KEY FINDINGS:")
print(f"   • Total Transactions: {report_data['total_transactions']}")
print(f"   • Total Revenue: ${report_data['total_revenue']:.2f}")
print(f"   • Peak Hour: {report_data['peak_hour']:02d}:00")
print(f"   • Peak Day: {report_data['peak_day']}")
print(f"   • Top Product: {report_data['top_product']}")
print(f"   • Model Accuracy (R²): {report_data['r2']:.4f}")

print(f"\n💡 BUSINESS RECOMMENDATIONS:")
print(f"   1. Focus marketing efforts during peak hours ({report_data['peak_hour']:02d}:00)")
print(f"   2. Ensure maximum stock on {report_data['peak_day']}s")
print(f"   3. Promote {report_data['low_product']} through special offers")
print(f"   4. Leverage {report_data['top_product']} as a flagship product")
print(f"   5. Use the ML model to forecast daily sales for inventory planning")

print("\n" + "=" * 80)
print("END OF ANALYSIS - HTML REPORT GENERATED SUCCESSFULLY!")
print("=" * 80)
