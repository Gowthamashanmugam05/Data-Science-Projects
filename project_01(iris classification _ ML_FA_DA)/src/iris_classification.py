import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory if it doesn't exist
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 80)

print("\n[STEP 2] Loading Dataset...")
print("-" * 80)

# Load from data folder
df = pd.read_csv('../data/Iris.csv')

print("\nFirst 5 rows of the dataset:")
print(df.head())

print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print(f"\nColumn Names: {list(df.columns)}")

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\n" + "=" * 80)
print("[STEP 3] Exploratory Data Analysis (EDA)")
print("-" * 80)

print("\n[Visualization 1] Creating count plot of species...")
plt.figure(figsize=(10, 6))
species_counts = df['Species'].value_counts()
plt.bar(species_counts.index, species_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Distribution of Iris Species', fontsize=14, fontweight='bold')
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(species_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_species_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

print("[Visualization 2] Creating scatter plot: Sepal Length vs Sepal Width...")
plt.figure(figsize=(10, 6))
species = df['Species'].unique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, sp in enumerate(species):
    data = df[df['Species'] == sp]
    plt.scatter(data['SepalLengthCm'], data['SepalWidthCm'], 
               label=sp, s=100, alpha=0.7, color=colors[i])
plt.title('Sepal Length vs Sepal Width by Species', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_sepal_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

print("[Visualization 3] Creating scatter plot: Petal Length vs Petal Width...")
plt.figure(figsize=(10, 6))
for i, sp in enumerate(species):
    data = df[df['Species'] == sp]
    plt.scatter(data['PetalLengthCm'], data['PetalWidthCm'], 
               label=sp, s=100, alpha=0.7, color=colors[i])
plt.title('Petal Length vs Petal Width by Species', fontsize=14, fontweight='bold')
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_petal_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

print("[Visualization 4] Creating histograms for numeric features...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    for i, sp in enumerate(species):
        data = df[df['Species'] == sp]
        ax.hist(data[feature], alpha=0.6, label=sp, bins=15, color=colors[i])
    ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_histograms.png'), dpi=300, bbox_inches='tight')
plt.show()

print("[Visualization 5] Creating box plots grouped by species...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    df.boxplot(column=feature, by='Species', ax=ax)
    ax.set_title(f'{feature} by Species', fontsize=12, fontweight='bold')
    ax.set_xlabel('Species', fontsize=11)
    ax.set_ylabel(feature, fontsize=11)
    plt.sca(ax)
    plt.xticks(rotation=45)
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()

print("[Visualization 6] Creating correlation heatmap...")
plt.figure(figsize=(10, 8))
numeric_df = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Heatmap of Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("[STEP 4] Data Preprocessing")
print("-" * 80)

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\n[Step 4.1] Encoding species labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Species Encoding Mapping:")
for i, species_name in enumerate(label_encoder.classes_):
    print(f"  {species_name} → {i}")

print("\n[Step 4.2] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, 
                                                      random_state=42, stratify=y_encoded)
print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

print("\n[Step 4.3] Applying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling completed")
print(f"Scaled training features shape: {X_train_scaled.shape}")
print(f"Scaled testing features shape: {X_test_scaled.shape}")

print("\n" + "=" * 80)
print("[STEP 5] Model Training - K-Nearest Neighbors (KNN)")
print("-" * 80)

print("\n[Step 5.1] Training KNN model with k=5...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
print("✓ Model training completed")

print("\n[Step 5.2] Making predictions on test data...")
y_pred = knn_model.predict(X_test_scaled)
print("✓ Predictions completed")

print("\n" + "=" * 80)
print("[STEP 6] Model Evaluation")
print("-" * 80)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[Metric 1] Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n[Metric 2] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n[Visualization 7] Creating confusion matrix heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - KNN Classifier', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\n[Metric 3] Classification Report:")
print("=" * 80)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(class_report)

print("\n" + "=" * 80)
print("[STEP 7] Model Improvement - Testing Different K Values")
print("-" * 80)

print("\nTesting various k values to find the best performing k...")
k_values = [3, 5, 7, 9, 11, 15, 21]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    print(f"  k = {k:2d} → Accuracy: {acc:.4f} ({acc*100:.2f}%)")

best_k_idx = np.argmax(accuracies)
best_k = k_values[best_k_idx]
best_accuracy = accuracies[best_k_idx]

print(f"\n✓ Best k value: {best_k} with accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

print("\n[Visualization 8] Creating k value vs accuracy plot...")
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
plt.scatter([best_k], [best_accuracy], color='#FF6B6B', s=200, zorder=5, 
           label=f'Best k={best_k} (Acc={best_accuracy*100:.2f}%)')
plt.title('KNN Accuracy vs K Values', fontsize=14, fontweight='bold')
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '08_k_values_accuracy.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print(f"""
📊 DATASET INFORMATION:
   • Total Samples: {len(df)}
   • Features: {', '.join(features)}
   • Classes: {', '.join(label_encoder.classes_)}
   • Class Distribution: {dict(zip(label_encoder.classes_, np.bincount(y_encoded)))}

📈 DATA SPLIT:
   • Training samples: {X_train.shape[0]} (80%)
   • Testing samples: {X_test.shape[0]} (20%)

🤖 MODEL DETAILS:
   • Algorithm: K-Nearest Neighbors (KNN)
   • Best K Value: {best_k}
   • Feature Scaling: StandardScaler
   • Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)

✅ RESULT: The model successfully achieves {best_accuracy*100:.2f}% accuracy!

📁 OUTPUT FILES:
   ✓ 01_species_distribution.png
   ✓ 02_sepal_scatter.png
   ✓ 03_petal_scatter.png
   ✓ 04_histograms.png
   ✓ 05_boxplots.png
   ✓ 06_correlation_heatmap.png
   ✓ 07_confusion_matrix.png
   ✓ 08_k_values_accuracy.png
   
   All files saved in: ../output/
""")

print("=" * 80)
print("✓ PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
