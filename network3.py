import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

data = {
    'ID': list(range(1, 25)),
    'Vibration': [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.25,
        0.45,
        0.65,
        0.85,
    ],
    'Slip': [
        0.10,
        0.20,
        0.15,
        0.30,
        0.25,
        0.35,
        0.40,
        0.50,
        0.45,
        0.55,
        0.50,
        0.60,
        0.80,
        0.70,
        0.85,
        0.75,
        0.90,
        0.80,
        0.85,
        0.70,
        0.90,
        0.95,
        0.85,
        0.90,
    ],
    'SurfaceGrip': [
        0.80,
        0.70,
        0.90,
        0.80,
        0.75,
        0.85,
        0.90,
        0.70,
        0.80,
        0.75,
        0.85,
        0.80,
        0.40,
        0.35,
        0.45,
        0.30,
        0.40,
        0.35,
        0.45,
        0.30,
        0.50,
        0.40,
        0.30,
        0.35,
    ],
    'ObstacleRate': [
        0.10,
        0.20,
        0.15,
        0.10,
        0.20,
        0.10,
        0.15,
        0.40,
        0.35,
        0.45,
        0.30,
        0.40,
        0.75,
        0.80,
        0.70,
        0.85,
        0.80,
        0.90,
        0.75,
        0.85,
        0.80,
        0.90,
        0.95,
        0.85,
    ],
    'Stable': [1] * 12 + [0] * 12,
}

df = pd.DataFrame(data)

features = ['Vibration', 'Slip', 'SurfaceGrip', 'ObstacleRate']

X_train = df.iloc[:18][features]
y_train = df.iloc[:18]['Stable']

X_test = df.iloc[18:][features]
y_test = df.iloc[18:]['Stable']

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
}

results = []

for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, pos_label=0)
  rec = recall_score(y_test, y_pred, pos_label=0)
  f1 = f1_score(y_test, y_pred, pos_label=0)
  cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

  results.append({
      'Модель': name,
      'Дәлдік (Accuracy)': round(acc, 2),
      'Нақтылық (Precision)': round(prec, 2),
      'Толықтық (Recall)': round(rec, 2),
      'F1': round(f1, 2),
      'Қателер матрицасы': cm.tolist(),
  })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))