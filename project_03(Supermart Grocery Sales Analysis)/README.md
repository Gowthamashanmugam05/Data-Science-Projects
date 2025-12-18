# Supermart Grocery Sales – Retail Analytics & Sales Prediction

## 📌 Project Overview

This is a comprehensive **Data Analytics + Machine Learning** project that analyzes grocery sales data from Supermart to identify trends, understand customer behavior, and build predictive models for sales forecasting.

### 🎯 Project Objectives

- **Analyze** grocery sales data across categories, cities, and time periods
- **Explore** patterns and trends through comprehensive EDA
- **Engineer** date-based and categorical features for ML modeling
- **Build** a Linear Regression model to predict sales
- **Evaluate** model performance with statistical metrics
- **Generate** actionable business insights for decision-making

---

## 📂 Project Structure

```
supermart_sales_analysis/
│
├── data/
│   └── supermart_grocery_sales.csv          # Raw sales dataset
│
├── src/
│   └── supermart_sales_analysis.py          # Main analysis script
│
├── output/
│   ├── 01_EDA_Analysis.png                  # EDA visualizations
│   └── 02_Model_Visualization.png           # Model performance plots
│
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## 🛠 Tools & Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **IDE** | VS Code |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Statistical Analysis** | Scipy, NumPy |

---

## 📊 Dataset Description

### Source File
`supermart_grocery_sales.csv`

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| Order ID | String | Unique order identifier |
| Customer Name | String | Customer name |
| Order Date | Date | Date of order (DD-MM-YYYY) |
| Category | Categorical | Product category (Tech, Furniture, Office Supplies) |
| Sub Category | Categorical | Product subcategory |
| City | Categorical | Order city location |
| Region | Categorical | Geographic region |
| State | Categorical | State of order |
| Sales | Numeric | Order sales amount ($) |
| Discount | Numeric | Discount percentage applied |
| Profit | Numeric | Profit from order ($) |

### Dataset Statistics

- **Total Records**: Thousands of transactions
- **Time Period**: Multiple years
- **Geographic Coverage**: Multiple cities and regions
- **Categories**: Multiple product categories
- **Target Variable**: Sales (numeric)

---

## 📋 Analysis Steps

### Step 1 & 2: Environment Setup & Data Loading
- ✓ Import required libraries
- ✓ Load dataset using Pandas
- ✓ Display dataset shape, columns, and basic statistics
- ✓ Check data types and missing values

### Step 3: Data Cleaning & Preprocessing
- ✓ Remove missing values and duplicate records
- ✓ Convert Order Date to datetime format
- ✓ Extract temporal features: `month_no`, `Month`, `year`
- ✓ Validate data integrity

### Step 4: Encoding Categorical Variables
- ✓ Apply Label Encoding to categorical columns:
  - Category, Sub Category, City, Region, State, Month
- ✓ Preserve encoding mappings for interpretation

### Step 5: Exploratory Data Analysis (EDA)

#### 5.1 Category-Level Analysis
- **Bar Chart**: Total sales by category
- **Insight**: Identifies top-performing product categories

#### 5.2 Time-Based Analysis
- **Line Chart**: Monthly sales trend
- **Pie Chart**: Yearly sales distribution
- **Insight**: Identifies seasonal patterns and growth trends

#### 5.3 City-Level Analysis
- **Bar Chart**: Top 5 cities by sales
- **Insight**: Geographic performance hotspots

#### 5.4 Correlation Analysis
- **Heatmap**: Feature correlation matrix
- **Insight**: Identifies relationships between variables
- **Special Focus**: Correlation with target variable (Sales)

### Step 6: Feature Selection for ML

**Features (X):**
- Category (encoded)
- Sub Category (encoded)
- City (encoded)
- Region (encoded)
- State (encoded)
- month_no
- Discount
- Profit

**Target (y):**
- Sales

**Dropped Columns:**
- Order ID, Customer Name, Order Date, Month

### Step 7: Train-Test Split & Scaling

- **Split Ratio**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)
- **Scaling**: StandardScaler applied to normalize features
- **Result**: Scaled features ready for model training

### Step 8: Model Training

- **Algorithm**: Linear Regression
- **Training Data**: 80% of cleaned dataset
- **Method**: Ordinary Least Squares (OLS)
- **Output**: Trained model with coefficients

### Step 9: Model Evaluation

**Metrics Used:**
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Interpretable error in sales units
- **R² Score**: Coefficient of determination (variance explained)

**Interpretation:**
- R² > 0.7: Excellent model fit
- R² 0.5-0.7: Good model fit
- R² < 0.5: Limited predictive power

### Step 10: Model Visualization

**Plots Generated:**
1. **Actual vs Predicted Sales**: Scatter plot with perfect prediction line
2. **Residual Plot**: Prediction errors visualization
3. **Residuals Distribution**: Histogram of model errors
4. **Feature Importance**: Top coefficients ranked

### Step 11: Business Insights

**Key Findings:**
1. **Best Performing Categories**: Ranked by total sales
2. **High-Demand Cities**: Top geographic markets
3. **Seasonal Patterns**: Monthly and yearly sales trends
4. **Factor Impact**: Discount and profit correlation with sales
5. **Model Performance**: R², RMSE, and predictive capability
6. **Strategic Recommendations**: Actionable business strategies

### Step 12: Documentation

- README.md: Complete project documentation
- Code Comments: Detailed explanations in source code
- Visualization Labels: Clear titles and axis labels
- Console Output: Formatted insights and metrics

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone/Download Project**
   ```bash
   cd supermart_sales_analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Data File Exists**
   - Place `supermart_grocery_sales.csv` in the `data/` folder

4. **Create Output Folder**
   ```bash
   mkdir output
   ```

5. **Run Analysis Script**
   ```bash
   python src/supermart_sales_analysis.py
   ```

### Expected Output

✓ Console output with statistics and insights
✓ Generated visualization plots saved to `output/`
✓ Model evaluation metrics displayed
✓ Business recommendations printed

---

## 📊 Key Findings

### Sales Performance
- **Total Revenue**: Sum of all sales transactions
- **Average Transaction**: Mean sales per order
- **Peak Category**: Highest revenue-generating category
- **Top Cities**: Geographic concentration of sales

### Temporal Insights
- **Seasonal Trends**: Monthly fluctuations in sales
- **Yearly Growth**: Year-over-year sales comparison
- **Peak Months**: Identify high-season periods

### Correlations
- **Profit & Sales**: Strong positive relationship
- **Discount & Sales**: Weak/moderate relationship
- **Category & Sales**: Varies by product type

### Model Performance
- **R² Score**: Percentage of variance explained by the model
- **RMSE**: Average prediction error in currency units
- **Coefficient Analysis**: Feature importance ranking

---

## 🤖 Model Details

### Algorithm: Linear Regression

**Equation:**
```
Sales = Intercept + (β₁ × Category) + (β₂ × Sub_Category) + ... + (β₈ × Profit)
```

### Model Assumptions
- ✓ Linear relationship between features and target
- ✓ Independent observations
- ✓ Constant variance in residuals
- ✓ Normally distributed errors

### Strengths
- Simple and interpretable model
- Fast training and prediction
- Clear feature importance
- Good for baseline comparison

### Limitations
- May underfit complex non-linear relationships
- Assumes linear correlations
- Sensitive to outliers
- May benefit from feature engineering

---

## 📈 Business Recommendations

1. **Category Focus**: Allocate resources to top-performing categories
2. **Geographic Expansion**: Expand operations in high-demand cities
3. **Inventory Management**: Implement seasonal stocking strategies
4. **Pricing Strategy**: Balance discounts vs. profit margins
5. **Predictive Planning**: Use model for demand forecasting
6. **Performance Monitoring**: Regular model retraining with new data

---

## 🔍 Troubleshooting

### Issue: "CSV File Not Found"
**Solution**: Ensure `supermart_grocery_sales.csv` is in the `data/` folder

### Issue: "Module Not Found"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "Permission Denied" (Linux/Mac)
**Solution**: Run `chmod +x src/supermart_sales_analysis.py`

### Issue: Plots Not Displaying
**Solution**: Update matplotlib backend or save plots to file instead

---

## 📚 Learning Outcomes

By completing this project, you'll learn:

- ✓ Data loading and exploratory analysis (EDA)
- ✓ Feature engineering from date columns
- ✓ Categorical variable encoding
- ✓ Train-test splitting methodology
- ✓ Feature scaling and normalization
- ✓ Linear Regression modeling
- ✓ Model evaluation metrics
- ✓ Data visualization techniques
- ✓ Business insight generation
- ✓ Professional code documentation

---

## 📝 Project Conclusion

This Supermart Grocery Sales Analytics project demonstrates a complete data science workflow from raw data exploration to machine learning model deployment. The Linear Regression model provides a baseline for sales prediction and identifies key drivers of revenue.

**Key Takeaway**: Data-driven insights combined with statistical modeling enable organizations to make informed business decisions and optimize operations for maximum profitability.

---

## 👨‍💼 Author Information

**Project Type**: Retail Analytics & Sales Prediction  
**Date Created**: 2025  
**Status**: ✅ Complete & Production-Ready

---

## 📞 Support & Feedback

For issues, improvements, or questions:
1. Review the code comments for detailed explanations
2. Check the console output for error messages
3. Verify dataset format and file paths
4. Ensure all dependencies are installed correctly

---

**Happy Analyzing! 📊✨**
