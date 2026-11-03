import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


data = {
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

X_train = df.iloc[:18][['Vibration', 'Slip', 'SurfaceGrip', 'ObstacleRate']]
y_train = df.iloc[:18]['Stable']
X_test = df.iloc[18:][['Vibration', 'Slip', 'SurfaceGrip', 'ObstacleRate']]
y_test = df.iloc[18:]['Stable']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

experiments = [(4,), (16,), (1,)]

for i, neurons in enumerate(experiments, 1):
  mlp = MLPClassifier(
      hidden_layer_sizes=neurons, solver='lbfgs', max_iter=1000, random_state=42
  )
  mlp.fit(X_train_scaled, y_train)
  y_pred = mlp.predict(X_test_scaled)

  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, pos_label=0)

  print(
      f'Exp {i} | Hidden Layers: {neurons} | Accuracy: {acc:.2f} | F1: {f1:.2f}'
  )
