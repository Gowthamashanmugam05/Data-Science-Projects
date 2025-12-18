# Coffee Sales Analysis & Forecasting Project

## 📌 Project Overview

This is a **Data Analyst** project focused on analyzing coffee vending machine sales data to understand sales patterns, customer behavior, and build a predictive model for sales forecasting.

### 🎯 Project Objectives

- Understand sales trends over time (monthly, daily, hourly patterns)
- Identify the most and least popular coffee products
- Analyze customer purchase behavior (payment methods, timing)
- Perform time-series based sales analysis
- Build a machine learning model to predict sales values
- Generate actionable business insights and recommendations

**Note:** This project is analysis-focused, not deployment-focused.

---

## 📂 Project Structure

```
coffee_sales_analysis/
│
├── data/
│   └── coffee_sales.csv              # Raw coffee sales dataset
│
├── src/
│   └── coffee_sales_analysis.py      # Main analysis script
│
├── requirements.txt                   # Python dependencies
│
└── README.md                          # This file
```

---

## 🛠 Tools & Technologies

| Component | Details |
|-----------|---------|
| **Language** | Python 3.x |
| **IDE** | Visual Studio Code |
| **Data Processing** | pandas, numpy |
| **Data Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn (Linear Regression) |
| **Format** | Jupyter-style Analysis Script |

---

## 📥 Installation & Setup

### Step 1: Create Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🚀 Running the Analysis

### Execute the Analysis Script

```bash
cd src/
python coffee_sales_analysis.py
```

### Output

The script will generate:
1. **Console Output**: Step-by-step analysis results and insights
2. **Visualizations**: 
   - `revenue_analysis.png` - Revenue and transaction analysis
   - `time_based_analysis.png` - Time series analysis
   - `hourly_by_product.png` - Hourly trends by product
   - `model_evaluation.png` - ML model performance charts

---

## 📊 Dataset Description

### File: `coffee_sales.csv`

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| date | Date | Date of transaction (YYYY-MM-DD) |
| datetime | DateTime | Full timestamp with seconds |
| cash_type | String | Payment method: "cash" or "card" |
| card | String | Anonymized card ID (ANON-xxxx-xxxx-xxxx) or empty for cash |
| money | Float | Sales amount in currency units |
| coffee_name | String | Type of coffee product |

**Sample Data:**
```
date        datetime                  cash_type  card                 money  coffee_name
2024-03-01  2024-03-01 10:15:50.520  card       ANON-0000-0000-0001  38.7   Latte
2024-03-01  2024-03-01 12:19:22.539  card       ANON-0000-0000-0002  38.7   Hot Chocolate
2024-03-02  2024-03-02 10:30:35.668  cash                            40.0   Latte
```

**Key Statistics:**
- **Total Records:** 1,135 transactions
- **Date Range:** March 2024 onwards
- **Coffee Products:** 6 types (Latte, Americano, Hot Chocolate, Cocoa, Espresso, Americano with Milk)
- **Payment Methods:** Card, Cash

---

## 📋 Analysis Steps Performed

### STEP 1: Environment Setup ✅
- Python virtual environment configured
- Required libraries installed

### STEP 2: Load Dataset ✅
- Loaded `coffee_sales.csv` using pandas
- Displayed dataset shape, columns, and data types
- Verified dataset integrity (1,135 rows × 6 columns)

### STEP 3: Data Cleaning ✅
- **Missing Values:** 
  - Identified missing values in 'card' column for cash transactions
  - Filled with 'CASH' label
- **Duplicates:** Checked and removed (if any)
- **Type Conversion:**
  - Converted 'date' column to datetime format
  - Converted 'datetime' column to datetime format

### STEP 4: Feature Engineering ✅
Created new features from existing data:
- **month**: Extracted YYYY-MM format (e.g., 2024-03)
- **day**: Weekday number (0=Monday, 6=Sunday)
- **hour**: Hour of transaction (0-23)
- **day_name**: Full day name (Monday, Tuesday, etc.)

These features enable time-based analysis and temporal pattern detection.

### STEP 5: Exploratory Data Analysis (EDA) ✅

#### 5.1 Transaction Analysis
- Total number of transactions analyzed
- Payment method distribution (cash vs card ratio)
- Top 5 most popular coffee products
- Transaction frequency by product

#### 5.2 Revenue Analysis
- Total revenue calculation
- Revenue breakdown by coffee product
- Identification of high and low-performing products
- Average revenue per transaction
- **Visualization:** Bar charts showing revenue and transaction counts

#### 5.3 Time-Based Analysis
- **Monthly Trend**: Line chart showing sales across months
- **Day-wise Sales**: Bar chart comparing revenue by weekday
- **Hour-wise Distribution**: Sales breakdown by hour of day
- **Hourly by Product**: Individual trend lines for each coffee product
- **Visualizations:** Multiple time-series plots for pattern recognition

### STEP 6: Sales Trend Insights ✅
Key findings extracted from analysis:
- Peak sales hours and days
- Top-selling and low-performing products
- Average transaction values
- Temporal patterns and seasonality
- **Printed Insights:** Console output with actionable findings

### STEP 7: Machine Learning Model ✅

#### 7.1 Data Preparation
- **Features Used:** hour, day, cash_type (encoded), coffee_name (encoded)
- **Target Variable:** money (sales amount)
- **Encoding:** 
  - LabelEncoder applied to categorical variables
  - Preserved original labels for interpretation
- **Data Split:** 80% training, 20% testing (stratified)

#### 7.2 Model Training
- **Algorithm:** Linear Regression
- **Reason:** Beginner-friendly, interpretable coefficients, good baseline model
- **Training:** Model fitted on 80% of data

#### 7.3 Model Evaluation
- **Predictions:** Generated predictions on test set
- **Metrics:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score (Coefficient of Determination)
- **Visualization:** Actual vs Predicted scatter plot + Residual plot

### STEP 8: Model Interpretation ✅
- Model coefficients for each feature
- Feature importance ranking
- Coefficient sign interpretation (positive/negative impact)
- R² score explanation
- Key insights on which features most influence sales

### STEP 9: Documentation ✅
- Comprehensive README.md created
- Code comments and section headers
- Dataset description and structure
- Analysis steps and methodology
- Business recommendations
- Model interpretation

---

## 🔍 Key Findings & Insights

### Sales Performance
- **Total Transactions:** 1,135 sales recorded
- **Total Revenue:** [Calculated in script output]
- **Average Transaction Value:** [Calculated in script output]

### Temporal Patterns
- **Peak Sales Hour:** Identified through hourly analysis
- **Peak Sales Day:** Most transactions occur on specific weekday
- **Monthly Trend:** Shows seasonal patterns or growth trends
- **Off-Peak Hours:** Identified for optimization opportunities

### Product Performance
- **Top Product:** Best-selling coffee with highest revenue
- **Low Performer:** Product with lowest sales - requires attention
- **Product Mix:** Distribution of sales across 6 coffee types
- **Opportunity:** Underperforming products need promotional strategy

### Payment Methods
- **Card vs Cash:** Distribution analysis
- **Customer Preference:** Dominant payment method
- **Average by Method:** Transaction value differences

---

## 🤖 Machine Learning Model Results

### Model Type: Linear Regression

**Performance Metrics:**
- Mean Squared Error (MSE): [From script output]
- Root Mean Squared Error (RMSE): [From script output]
- R² Score: [From script output]

**Model Coefficients:**
- hour: [Coefficient value]
- day: [Coefficient value]
- cash_type: [Coefficient value]
- coffee_name: [Coefficient value]

**Interpretation:**
- Positive coefficient: Feature increases predicted sales
- Negative coefficient: Feature decreases predicted sales
- R² close to 1: Model explains most of the sales variation
- R² close to 0: Model has limited predictive power

**Model Strengths:**
- Interpretable results (understand feature impacts)
- Fast training and prediction
- Good baseline for comparison with advanced models
- Captures linear relationships

**Model Limitations:**
- Assumes linear relationships between features and sales
- May miss complex non-linear patterns
- Doesn't capture interactions between features
- Affected by outliers in data

---

## 💡 Business Recommendations

Based on the analysis, the following recommendations are proposed:

1. **Maximize Peak Hours**
   - Focus marketing and customer engagement during identified peak hours
   - Ensure adequate staffing during high-traffic periods
   - Consider promotional offers during off-peak hours

2. **Inventory Management**
   - Increase stock of top-selling products before peak days
   - Plan for expected demand based on temporal patterns
   - Use ML model for inventory forecasting

3. **Product Strategy**
   - Promote low-performing products through special offers or discounts
   - Leverage top-selling products as revenue drivers
   - Test new product combinations during peak hours

4. **Payment Method Optimization**
   - Support dominant payment method with excellent infrastructure
   - Consider incentives for alternative payment methods if needed

5. **Data-Driven Operations**
   - Schedule staff based on hourly/daily patterns
   - Plan maintenance during low-traffic periods
   - Use model predictions for short-term sales forecasting

6. **Future Enhancements**
   - Collect more granular customer data for segmentation
   - Analyze seasonal variations across years
   - Implement advanced ML models (Random Forest, XGBoost)
   - Consider external factors (weather, events, holidays)

---

## 📈 Visualizations Generated

### 1. revenue_analysis.png
- **Top Left:** Total revenue by coffee product (horizontal bar chart)
- **Top Right:** Number of transactions by coffee product (horizontal bar chart)

### 2. time_based_analysis.png
- **Top Left:** Monthly sales trend (line chart with markers)
- **Top Right:** Day-wise sales comparison (vertical bar chart)
- **Bottom Left:** Hour-wise sales distribution (vertical bar chart)
- **Bottom Right:** Hour-wise transaction count (vertical bar chart)

### 3. hourly_by_product.png
- Individual line plots for each coffee product
- Shows hourly revenue patterns specific to each product
- Helps identify product-specific peak hours

### 4. model_evaluation.png
- **Left:** Actual vs Predicted sales (scatter plot with perfect prediction line)
- **Right:** Residual plot (prediction errors analysis)

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError for CSV
**Solution:** Verify the data file path in the script matches your directory structure
```python
df = pd.read_csv('../data/coffee_sales.csv')
```

### Issue: Warnings about deprecation
**Solution:** Warnings are suppressed in the script with `warnings.filterwarnings('ignore')`

### Issue: Charts not displaying
**Solution:** Charts are saved to PNG files instead of displaying. Check the `data/` folder for generated images.

---

## 📚 Future Enhancements

1. **Advanced ML Models**
   - Random Forest for non-linear patterns
   - XGBoost for improved accuracy
   - Time series models (ARIMA) for forecasting

2. **Feature Engineering**
   - Lag features (sales from previous hours)
   - Rolling averages for trend analysis
   - Seasonal decomposition
   - External features (weather, holidays)

3. **Dashboard Development**
   - Interactive dashboards using Plotly/Streamlit
   - Real-time sales monitoring
   - Predictive alerts for unusual patterns

4. **Customer Insights**
   - Customer segmentation analysis
   - Loyalty program impact assessment
   - Churn analysis for card-based customers

5. **Operational Optimization**
   - Resource allocation based on predictions
   - Pricing optimization strategies
   - Cross-sell recommendations

---

## 📄 Project Conclusion

This project successfully demonstrates:
- ✅ Data loading and cleaning practices
- ✅ Feature engineering for temporal analysis
- ✅ Comprehensive exploratory data analysis
- ✅ Data visualization techniques
- ✅ Machine learning model development and evaluation
- ✅ Business insights generation
- ✅ Professional documentation

The analysis provides a solid foundation for data-driven decision-making in coffee sales operations and serves as a template for similar retail analytics projects.

---

## 📞 Contact & Support

For questions or improvements to this project:
- Refer to inline code comments in `coffee_sales_analysis.py`
- Check the console output for detailed step-by-step results
- Review visualization files for graphical insights

---

## 📝 License

This project is created for educational purposes.

---

**Last Updated:** December 2024  
**Version:** 1.0  
**Status:** ✅ Complete
