import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

# Función para guardar resultados en JSON
def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f)

# Función para cargar resultados anteriores
def load_previous_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return None

# Cargar los datos desde SQLite
dataset = "dataset_2012-24"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Asegurarse de que la variable objetivo 'Home-Team-Win' sea binaria
margin = data['Home-Team-Win'].apply(lambda x: 1 if x > 0 else 0)

# Eliminar las columnas innecesarias
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)
data = data.astype(float)

# Dividir los datos
x_train, x_test, y_train, y_test = train_test_split(data.values, margin, test_size=0.1, random_state=42)

# Parámetros del modelo
#param = {
#    'max_depth': 5,                # Ajusta según la complejidad
#    'learning_rate': 0.02,         # Disminuye la tasa de aprendizaje para un ajuste más preciso
#    #'n_estimators': 1000,          # Aumenta el número de árboles
#    'subsample': 0.8,              # Fracción de datos usados para cada árbol
#    'colsample_bytree': 0.8,       # Fracción de características usadas para cada árbol
#    'objective': 'multi:softprob', # Clasificación multiclase
#    'num_class': 2,                # Número de clases (victoria/derrota)
#    'lambda': 0.5,                   # Regularización L2
#    'alpha': 0                   # Regularización L1
#}
#epochs = 500 #65.7


# Parámetros del modelo
model = XGBClassifier(
    max_depth=5,                    # Reducimos la profundidad para evitar sobreajuste
    learning_rate=0.02,              # Incrementamos el learning_rate para un aprendizaje más rápido
    n_estimators=850,               # Reducimos los estimadores para equilibrar con el learning_rate
    subsample=0.8,                  # Fracción de datos usados para cada árbol
    colsample_bytree=0.8,           # Fracción de características usadas para cada árbol
    objective='binary:logistic',    # Clasificación binaria
    reg_lambda=1.5,                   # Ajustamos regularización L2 para evitar sobreajuste
    reg_alpha=1                  # Ajustamos regularización L1
)

# Validación cruzada para evaluar el modelo antes de probarlo en el conjunto de prueba
print("Realizando validación cruzada...")
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy')
print(f"Precisión promedio en validación cruzada: {np.mean(cv_scores) * 100:.2f}%")


# Entrenar el modelo
#train = xgb.DMatrix(x_train, label=y_train)
#test = xgb.DMatrix(x_test, label=y_test)

# Entrenamiento del modelo
#model = xgb.train(param, train, epochs)

# Entrenamiento del modelo XGBClassifier
#model.fit(x_train, y_train)
model.fit(
    x_train, y_train, 
    eval_set=[(x_test, y_test)], 
    verbose=True
)

# Mostrar la importancia de las características después del entrenamiento
#xgb.plot_importance(model)
#plt.show()

# Predecir y calcular la precisión
predictions = model.predict(x_test)
acc = round(accuracy_score(y_test, predictions) * 100, 1)
print(f"Precisión: {acc}%")

# Guardar el modelo si es la mejor precisión hasta el momento
results_file = '../../Models/xgboost_training_results.json'
previous_results = load_previous_results(results_file)

best_accuracy = 0
if previous_results:
    best_accuracy = previous_results['best_accuracy']

if acc > best_accuracy:
    model.save_model(f'../../Models/XGBoost_{acc}%_ML-4.json')
    results = {
        'best_accuracy': acc
    }
    save_results(results, results_file)
    print(f"Nuevo mejor modelo guardado con precisión de {acc}%")
