import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Деректер жиынтығын құру (24 жол)
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

# 2. Оқыту (ID 1–18) және Тестілеу (ID 19–24) жиындарына бөлу
features = ['Vibration', 'Slip', 'SurfaceGrip', 'ObstacleRate']

X_train = df.iloc[:18][features]
y_train = df.iloc[:18]['Stable']

X_test = df.iloc[18:][features]
y_test = df.iloc[18:]['Stable']
test_ids = df.iloc[18:]['ID'].values

# 3. Алдын ала өңдеу (Нормализация / Масштабтау)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Базалық модель (Decision Tree)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 5. Нейрондық желі (MLPClassifier)
nn_model = MLPClassifier(
    hidden_layer_sizes=(8,), solver='lbfgs', max_iter=1000, random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)

# 6. Метрикалар мен қателер матрицасын басып шығару
print('--- Базалық модель (Decision Tree) ---')
print('Accuracy:', accuracy_score(y_test, dt_preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, dt_preds, labels=[1, 0]))

print('\n--- Нейрондық желі (MLPClassifier) ---')
print('Accuracy:', accuracy_score(y_test, nn_preds))
print('Confusion Matrix:\n', confusion_matrix(y_test, nn_preds, labels=[1, 0]))

# 7. Тест ID (19–24) бойынша қателерді талдау кестесін шығару
results = []
for idx, t_id, actual, dt_p, nn_p in zip(
    range(len(test_ids)), test_ids, y_test, dt_preds, nn_preds
):
  is_correct = (dt_p == actual) and (nn_p == actual)
  status = 'Дұрыс' if is_correct else 'Қате'
  reason = (
      '—'
      if is_correct
      else 'Деректердің дисбалансы немесе өңделмеуі салдарынан қате болжам жасалды'
  )
  results.append({
      'Тест ID': t_id,
      'Нақты класс': actual,
      'Базалық модель болжамы': dt_p,
      'Нейрондық желі болжамы': nn_nn_p if 'nn_nn_p' in locals() else nn_p,
      'Дұрыс па?': status,
      'Қате болса, ықтимал себебі': reason,
  })

results_df = pd.DataFrame(results)
print('\n--- Тест ID (19–24) бойынша нәтижелер ---')
print(results_df.to_string(index=False))