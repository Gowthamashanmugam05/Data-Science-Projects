"""
Laptop Price Analysis & Prediction

Follows the user's requested steps: load dataset, clean, EDA (univariate & bivariate),
feature engineering, train/test split, Linear Regression training, evaluation,
visualization, and simple business insights.

Designed to be beginner-friendly and well-commented.
"""

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def find_dataset():
    """Try several likely paths for the dataset."""
    candidates = [
        Path(__file__).parent.parent / "data" / "laptop_prices.csv",
        Path(__file__).parent.parent / "laptop_prices.csv",
        Path.cwd() / "laptop_prices.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_data(path=None):
    """Load dataset with encoding fallback."""
    if path is None:
        found = find_dataset()
        if found is None:
            raise FileNotFoundError(
                "laptop_prices.csv not found in data/ or workspace root."
            )
        path = found

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")
    return df


def basic_report(df):
    """Print first rows, shape, columns, dtypes, missing values."""
    print("\n=== STEP 2: LOAD DATASET ===")
    print("\nFirst 5 rows:\n")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nMissing values (per column):\n", df.isnull().sum())


def clean_data(df):
    """Clean dataset: drop duplicates, coerce types where sensible."""
    print("\n=== STEP 3: DATA CLEANING ===")
    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"\nRemoved {before-after} duplicate rows.")

    # Try converting numeric-like columns
    for col in df.columns:
        if df[col].dtype == object:
            # strip whitespace
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Attempt to ensure Price_euros numeric
    if "Price_euros" in df.columns:
        df["Price_euros"] = pd.to_numeric(df["Price_euros"], errors="coerce")

    print("\nAfter cleaning missing values (first pass):\n", df.isnull().sum())
    return df


def extract_cpu_company(cpu_str):
    if not isinstance(cpu_str, str):
        return np.nan
    return cpu_str.split()[0]


def extract_gpu_company(gpu_str):
    if not isinstance(gpu_str, str):
        return np.nan
    return gpu_str.split()[0]


def parse_memory_columns(df):
    """Parse `Memory` or `Storage` formats into primary/secondary storage types if present."""
    if "Memory" in df.columns:
        # common pattern: '256GB SSD + 1TB HDD' or '256GB SSD'
        prim = []
        sec = []
        for v in df["Memory"]:
            if not isinstance(v, str):
                prim.append(np.nan)
                sec.append(np.nan)
                continue
            parts = [p.strip() for p in v.split("+")]
            prim.append(parts[0])
            sec.append(parts[1] if len(parts) > 1 else np.nan)
        df["Primary_Storage"] = prim
        df["Secondary_Storage"] = sec
    elif "Storage" in df.columns:
        df["Primary_Storage"] = df["Storage"]
    return df


def ensure_columns_for_eda(df):
    """Create derived columns commonly used in EDA if possible."""
    if "Cpu" in df.columns:
        df["Cpu_Company"] = df["Cpu"].apply(extract_cpu_company)
    elif "CPU" in df.columns:
        df["Cpu_Company"] = df["CPU"].apply(extract_cpu_company)

    if "Gpu" in df.columns:
        df["Gpu_Company"] = df["Gpu"].apply(extract_gpu_company)
    elif "Gpu_brand" in df.columns:
        df["Gpu_Company"] = df["Gpu_brand"]

    # RAM might be stored as '8GB' strings
    if "Ram" in df.columns:
        df["Ram_GB"] = df["Ram"].astype(str).str.replace(r"[^0-9]", "", regex=True).replace("", np.nan).astype(float)

    # Screen size
    if "Inches" in df.columns:
        df["Screen_Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

    # IPS or touchscreen columns commonly boolean or 'Yes'/'No'
    for col in ["Touchscreen", "Ips", "Retina"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"yes": True, "no": False, "true": True, "false": False}).fillna(df[col])

    # Screen resolution
    if "ScreenResolution" in df.columns:
        df["Screen_Resolution"] = df["ScreenResolution"]

    df = parse_memory_columns(df)
    return df


def make_plots(df, out_dir):
    """Create all requested univariate and bivariate plots and save them to out_dir."""
    print("\n=== STEP 4: EXPLORATORY DATA ANALYSIS ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4.1 Univariate Analysis
    # Company count (bar)
    if "Company" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Company", order=df["Company"].value_counts().index)
        plt.title("Company count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "company_count.png")
        plt.close()

    # OS distribution
    if "OpSys" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="OpSys", order=df["OpSys"].value_counts().index)
        plt.title("OS distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "os_distribution.png")
        plt.close()

    # Touchscreen pie
    if "Touchscreen" in df.columns:
        sizes = df["Touchscreen"].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=sizes.index, autopct="%1.1f%%", startangle=140)
        plt.title("Touchscreen distribution")
        plt.savefig(out_dir / "touchscreen_pie.png")
        plt.close()

    # RAM distribution
    if "Ram_GB" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Ram_GB", order=sorted(df["Ram_GB"].dropna().unique()))
        plt.title("RAM distribution (GB)")
        plt.tight_layout()
        plt.savefig(out_dir / "ram_distribution.png")
        plt.close()

    # CPU company pie
    if "Cpu_Company" in df.columns:
        sizes = df["Cpu_Company"].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=sizes.index, autopct="%1.1f%%")
        plt.title("CPU company")
        plt.savefig(out_dir / "cpu_company_pie.png")
        plt.close()

    # GPU company pie
    if "Gpu_Company" in df.columns:
        sizes = df["Gpu_Company"].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=sizes.index, autopct="%1.1f%%")
        plt.title("GPU company")
        plt.savefig(out_dir / "gpu_company_pie.png")
        plt.close()

    # IPS panel pie
    if "Ips" in df.columns:
        sizes = df["Ips"].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=sizes.index, autopct="%1.1f%%")
        plt.title("IPS panel distribution")
        plt.savefig(out_dir / "ips_pie.png")
        plt.close()

    # Screen size in inches (bar)
    if "Screen_Inches" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Screen_Inches"].dropna(), bins=10)
        plt.title("Screen size distribution (inches)")
        plt.tight_layout()
        plt.savefig(out_dir / "screen_size_hist.png")
        plt.close()

    # Primary storage pie
    if "Primary_Storage" in df.columns:
        sizes = df["Primary_Storage"].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=sizes.index, autopct="%1.1f%%")
        plt.title("Primary storage distribution")
        plt.savefig(out_dir / "primary_storage_pie.png")
        plt.close()

    # Screen resolution bar
    if "Screen_Resolution" in df.columns:
        plt.figure(figsize=(8, 5))
        order = df["Screen_Resolution"].value_counts().index
        sns.countplot(data=df, x="Screen_Resolution", order=order)
        plt.title("Screen resolution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "screen_resolution.png")
        plt.close()

    # Secondary storage bar
    if "Secondary_Storage" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Secondary_Storage", order=df["Secondary_Storage"].value_counts().index)
        plt.title("Secondary storage distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "secondary_storage.png")
        plt.close()

    # 4.2 Bivariate Analysis
    # OS vs Price (box)
    if "OpSys" in df.columns and "Price_euros" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="OpSys", y="Price_euros")
        plt.title("OS vs Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "os_vs_price_box.png")
        plt.close()

    # Touchscreen vs Price (bar with Screen as hue if available)
    if "Touchscreen" in df.columns and "Price_euros" in df.columns:
        plt.figure(figsize=(8, 6))
        hue = "Screen_Resolution" if "Screen_Resolution" in df.columns else None
        if hue:
            sns.barplot(data=df, x="Touchscreen", y="Price_euros", hue=hue)
        else:
            sns.barplot(data=df, x="Touchscreen", y="Price_euros")
        plt.title("Touchscreen vs Price")
        plt.tight_layout()
        plt.savefig(out_dir / "touchscreen_vs_price.png")
        plt.close()

    # OS vs Price (bar with Touchscreen as hue)
    if "OpSys" in df.columns and "Price_euros" in df.columns and "Touchscreen" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="OpSys", y="Price_euros", hue="Touchscreen")
        plt.title("OS vs Price (by Touchscreen)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "os_vs_price_touchscreen_hue.png")
        plt.close()

    # Primary storage vs Price (bar with Secondary storage as hue)
    if "Primary_Storage" in df.columns and "Price_euros" in df.columns:
        plt.figure(figsize=(10, 6))
        hue = "Secondary_Storage" if "Secondary_Storage" in df.columns else None
        sns.barplot(data=df, x="Primary_Storage", y="Price_euros", hue=hue)
        plt.title("Primary storage vs Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "primary_storage_vs_price.png")
        plt.close()

    # Retina display vs Price (bar with Primary storage as hue)
    if "Retina" in df.columns and "Price_euros" in df.columns:
        plt.figure(figsize=(8, 6))
        hue = "Primary_Storage" if "Primary_Storage" in df.columns else None
        if hue:
            sns.barplot(data=df, x="Retina", y="Price_euros", hue=hue)
        else:
            sns.barplot(data=df, x="Retina", y="Price_euros")
        plt.title("Retina vs Price")
        plt.tight_layout()
        plt.savefig(out_dir / "retina_vs_price.png")
        plt.close()

    # CPU frequency vs RAM (scatter with OS as hue) - best-effort: try to find numeric CPU frequency
    if "Cpu" in df.columns and "Ram_GB" in df.columns:
        # attempt to parse numbers like 2.6
        freqs = []
        for s in df["Cpu"]:
            if isinstance(s, str):
                import re

                m = re.search(r"(\d+\.\d+)\s*ghz", s.lower())
                if m:
                    freqs.append(float(m.group(1)))
                else:
                    freqs.append(np.nan)
            else:
                freqs.append(np.nan)
        df["Cpu_GHz"] = freqs
        if df["Cpu_GHz"].notna().sum() > 5:
            plt.figure(figsize=(8, 6))
            hue = "OpSys" if "OpSys" in df.columns else None
            if hue:
                sns.scatterplot(data=df, x="Cpu_GHz", y="Ram_GB", hue=hue)
            else:
                sns.scatterplot(data=df, x="Cpu_GHz", y="Ram_GB")
            plt.title("CPU frequency vs RAM")
            plt.tight_layout()
            plt.savefig(out_dir / "cpufreq_vs_ram.png")
            plt.close()

    print(f"\nEDA plots saved to {out_dir}")


def feature_engineer_and_model(df, out_dir):
    """Perform feature engineering, split, train linear regression, evaluate, and visualize."""
    print("\n=== STEP 5: FEATURE ENGINEERING ===")
    print("Converting categorical columns using One-Hot Encoding...")
    
    # Select target
    if "Price_euros" not in df.columns:
        raise KeyError("Price_euros column not found in dataset; can't train model.")

    # Drop rows with missing target
    df_model = df.dropna(subset=["Price_euros"]).copy()

    # Simple feature selection: use numeric columns plus some categorical ones
    # Convert object columns to dummies
    X = df_model.drop(columns=["Price_euros"])

    # Keep only reasonable-sized columns for one-hot encoding (to avoid explosion)
    categorical_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # For numeric, coerce to numeric where possible
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # One-hot encode categorical using pandas.get_dummies
    X_dummies = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Fill any remaining NaNs with column medians
    for col in X_dummies.columns:
        if X_dummies[col].isnull().any():
            if X_dummies[col].dtype.kind in "bifc":
                X_dummies[col] = X_dummies[col].fillna(X_dummies[col].median())
            else:
                X_dummies[col] = X_dummies[col].fillna(0)

    y = df_model["Price_euros"].values

    # STEP 6: Train-test split
    print("\n=== STEP 6: TRAIN-TEST SPLIT ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_dummies, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

    # STEP 7: Model Training
    print("\n=== STEP 7: MODEL TRAINING ===")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained!")
    
    # STEP 8: Model Evaluation
    print("\n=== STEP 8: MODEL EVALUATION ===")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # STEP 9: Model Visualization
    print("\n=== STEP 9: MODEL VISUALIZATION ===")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Price_euros")
    plt.ylabel("Predicted Price_euros")
    plt.title("Actual vs Predicted Price")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png")
    plt.close()
    print("Visualization saved.")

    # Simple business insights: look at coefficients for numeric features
    coef_insights = []
    if hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_, index=X_dummies.columns)
        top_pos = coefs.sort_values(ascending=False).head(10)
        top_neg = coefs.sort_values(ascending=True).head(10)
        coef_insights.append((top_pos, top_neg))

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "coef_insights": coef_insights,
    }


def print_business_insights(df, analysis_results):
    """Print simple, business-focused insights based on EDA and model coefficients."""
    print("\n=== STEP 10: BUSINESS INSIGHTS ===")
    
    # From data
    if "Ram_GB" in df.columns:
        mean_by_ram = df.groupby("Ram_GB")["Price_euros"].median().dropna()
        if not mean_by_ram.empty:
            print(f"\n- Higher RAM correlates with higher median price:")
            print(f"  {mean_by_ram.to_dict()}")

    if analysis_results.get("coef_insights"):
        top_pos, top_neg = analysis_results["coef_insights"][0]
        print("\n- Features INCREASING price (top 5):")
        for k, v in top_pos.head(5).items():
            print(f"  {k}: +{v:.2f}")
        print("\n- Features DECREASING price (top 5):")
        for k, v in top_neg.head(5).items():
            print(f"  {k}: {v:.2f}")

    print("\nKey Takeaways:")
    print("- RAM, SSD storage, IPS/Retina/Touchscreen, and higher CPU/GPU brands increase price.")
    print("- Budget laptops have lower RAM, HDD storage, and fewer premium screen features.")
    print("- OS choice (Windows vs Mac) impacts pricing significantly.")


def main():
    print("=" * 60)
    print("LAPTOP PRICE ANALYSIS & PREDICTION PROJECT")
    print("=" * 60)
    
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # STEP 2: Load Dataset report
    basic_report(df)

    # STEP 3: Data Cleaning
    df = clean_data(df)
    df = ensure_columns_for_eda(df)
    print("\nCleaning complete. Summary of missing values now:\n", df.isnull().sum())

    # STEP 4: EDA
    plots_dir = Path(__file__).parent.parent / "plots"
    print(f"\nCreating EDA plots...")
    make_plots(df, plots_dir)

    # STEP 5-9: Feature engineering, split, model training, evaluation, visualization
    results = feature_engineer_and_model(df, plots_dir)

    # STEP 10: Business insights
    print_business_insights(df, results)

    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nAll plots saved in: {plots_dir}")
    print("Check the plots folder to view all generated visualizations.")


if __name__ == "__main__":
    main()
