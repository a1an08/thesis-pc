import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv("log.csv")

# 0 normal
df['correction'] = df['correction'].fillna(0).astype(int)

df = df.ffill().fillna(0)

print("Generating lag features...")
sensor_cols = [col for col in df.columns if col not in ['timestamp', 'relative_img_path', 'correction']]

for col in sensor_cols:
    df[f"{col}_lag1"] = df[col].shift(1)

df = df.dropna()

X = df.drop(columns=['timestamp', 'relative_img_path', 'correction'])
y = df['correction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


weights = compute_sample_weight(class_weight='balanced', y=y_train)

print("Training the model...")

num_classes = len(np.unique(y))

model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    max_depth=6,              # Depth of the decision trees
    learning_rate=0.1,        # Step size for learning
    n_estimators=150,         # Number of trees to build
    tree_method='hist'        # Highly efficient for large tabular data
)

model.fit(X_train, y_train, sample_weight=weights)

print("\n--- Model Evaluation ---")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# --- Confusion Matrix ---

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted State")
plt.ylabel("Actual State")
plt.show()

# --- Feature Importance ---

plt.figure(figsize=(10, 8))

xgb.plot_importance(model, max_num_features=15, importance_type='gain', title="Top 15 Most Important Sensors")
plt.show()

model_filename = "xgboost_printer_model.json"
model.save_model(model_filename)
print(f"\nModel successfully saved to {model_filename}")