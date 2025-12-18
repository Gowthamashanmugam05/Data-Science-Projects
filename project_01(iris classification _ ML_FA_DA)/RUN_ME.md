# 🌸 Iris Flower Classification - ML Project

## Quick Start

```bash
cd src
python iris_classification.py
```

## What's This Project?

A machine learning classification project using **K-Nearest Neighbors (KNN)** to classify iris flowers into 3 species based on their measurements.

### Results
- **Accuracy: 96.67%** ✅
- **Best K Value: 7**
- **Test Set: 30 samples**

## Project Structure

```
├── data/
│   └── Iris.csv              # Dataset (150 flowers, 3 species)
├── src/
│   └── iris_classification.py # Main script
├── output/                     # Generated visualizations
│   ├── 01_species_distribution.png
│   ├── 02_sepal_scatter.png
│   ├── 03_petal_scatter.png
│   ├── 04_histograms.png
│   ├── 05_boxplots.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_confusion_matrix.png
│   └── 08_k_values_accuracy.png
├── requirements.txt
└── README.md
```

## Steps in the Pipeline

1. **Load Dataset** - 150 iris flower samples
2. **Exploratory Data Analysis (EDA)** - 6 visualizations
3. **Data Preprocessing** - Encode labels, split data (80/20), scale features
4. **Model Training** - KNN with k=5
5. **Model Evaluation** - Accuracy, Confusion Matrix, Classification Report
6. **Model Improvement** - Test k values 3 to 21, find best k=7 (96.67% accuracy)

## Features Used

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## Iris Species

- **Iris-setosa** (Red)
- **Iris-versicolor** (Blue)
- **Iris-virginica** (Purple)

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install with: `pip install -r requirements.txt`

---

**Status:** ✅ Complete & Working
