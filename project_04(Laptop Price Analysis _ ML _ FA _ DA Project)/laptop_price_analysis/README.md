# Laptop Price Analysis & Prediction

Project to analyze laptop specifications and build a linear regression model to predict laptop prices (`Price_euros`).

## Overview
- Loads `laptop_prices.csv` and performs cleaning, exploratory data analysis (plots), simple feature engineering, trains a `LinearRegression` model, evaluates it, and prints business insights.

## Usage
1. Ensure Python 3.8+ is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset `laptop_prices.csv` inside the `data/` folder or workspace root. The script will try to find it.

4. Run the analysis:

```bash
python src/laptop_price_analysis.py
```

## Files
- `data/` - expected location for `laptop_prices.csv`
- `src/laptop_price_analysis.py` - main script performing the analysis and modelling.
- `requirements.txt` - Python dependencies.

## Notes
- The script includes checks for common column patterns and will attempt to be robust to small schema differences.
- All EDA plots are saved in a `plots/` folder created automatically.
