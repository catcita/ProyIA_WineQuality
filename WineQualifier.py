# Librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)

# 1. CARGA DE DATOS
#Este dataset les proporciona las características (X) (la acidez, el alcohol, etc.) y la variable
# objetivo (Y) (la calidad del vino), que es el pilar para el aprendizaje supervisado.
dataframe = pd.read_csv('winequality-white.csv', sep=';')

# 2. EXPLORACIÓN INICIAL
print(dataframe.head()) # se imprimen las primeras 5 filas del dataframe
print(dataframe.info()) # se imprimen los tipos de datos de cada columna y si hay nulos
print(dataframe.describe())
print(dataframe['quality'].value_counts().sort_index()) #Cuenta cuántos vinos hay para
                                                        # cada puntaje de calidad (de 3 a 9).

# 3. CONVERSIÓN A CLASIFICACIÓN BINARIA
dataframe['quality_label'] = (dataframe['quality'] > 6).astype(int) # crea una nueva columna 'quality_label'
print(f"\nDistribución de clases:")
print(dataframe['quality_label'].value_counts())
print(f"Porcentaje clase 1: {dataframe['quality_label'].mean() * 100:.2f}%")

# 4. PREPARACIÓN DE DATOS
X = dataframe.drop(['quality', 'quality_label'], axis=1) # Características de los vinos - calidad
y = dataframe['quality_label'] # Variable objetivo - calidad

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODELO BASELINE (sin optimización)
randomforest_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
randomforest_baseline.fit(X_train_scaled, y_train)

y_pred_baseline = randomforest_baseline.predict(X_test_scaled)
print(f"\n=== BASELINE RANDOM FOREST ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"\n{classification_report(y_test, y_pred_baseline)}")

# 6. OPTIMIZACIÓN DE HIPERPARÁMETROS
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

randomforest = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    randomforest, param_grid, cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores parámetros: {grid_search.best_params_}")
print(f"Mejor score CV: {grid_search.best_score_:.4f}")

# 7. EVALUACIÓN MODELO OPTIMIZADO
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

print(f"\n=== MODELO OPTIMIZADO ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# 8. CROSS-VALIDATION
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 9. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)