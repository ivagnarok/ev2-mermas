# IMPLEMENTACIÓN DE ANÁLISIS PREDICTIVO COMPLETO
# Utilizamos train.csv disponible en https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting (Es necesario 
# registrarse en la página de kaggle). Luego incorporar el archivo en el directorio de trabajo con python.
# Las librerias necesarias están en el archivo requirements.txt

# PASO 1: IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime as dt
import holidays
from sklearn.base import BaseEstimator, TransformerMixin

print("*IMPLEMENTACIÓN DE MODELOS PREDICTIVOS. CASO PREDICTIVO DE MERMAS*")

# PASO 2: CARGA Y PREPARACIÓN DE DATOS
df = pd.read_excel('mermas_actividad_unidad_2.xlsx')

df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
df = df.dropna(subset=['fecha'])

cl_holidays = holidays.country_holidays('CL', years=df['fecha'].dt.year.unique())
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['dia_semana'] = df['fecha'].dt.dayofweek
df['es_finde'] = df['dia_semana'] >= 5
df['es_feriado'] = df['fecha'].isin(cl_holidays)


df = df[df['merma_unidad_p'] > 0]
df = df[df['merma_unidad_p'] < df['merma_unidad_p'].quantile(0.98)]
df['log_merma'] = np.log1p(df['merma_unidad_p'])

cat_vars = ['descripcion','categoria','negocio','seccion']
for col in cat_vars:
    top = df[col].value_counts().nlargest(30).index
    df[col] = df[col].where(df[col].isin(top), 'OTROS')

df['media_mes_producto'] = df.groupby(['descripcion','mes'])['merma_unidad_p'].transform('mean')
df['ranking_categoria'] = df.groupby('categoria')['merma_unidad_p'].rank(ascending=False)

features =['año','mes','semana','dia_semana','es_finde','es_feriado','media_mes_producto','ranking_categoria'] + cat_vars
X = df[features]
y = df['log_merma']
y_real = df['merma_unidad_p']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['descripcion','categoria','negocio', 'seccion']
numeric_features = ['mes','año','semana','dia_semana','media_mes_producto','ranking_categoria']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

class CatBoostWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features):
        self.cat_features = cat_features
        self.model = CatBoostRegressor(verbose=0, random_state=42,
                                       learning_rate=0.03, depth=8,
                                       iterations=700, l2_leaf_reg=3)

    def fit(self, X, y):
        self.model.fit(X, y, cat_features=self.cat_features)
        return self

    def predict(self, X):
        return self.model.predict(X)

cat_features_idx = [X.columns.get_loc(c) for c in cat_vars]

pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline_cb = Pipeline([
    ('regressor', CatBoostWrapper(cat_features=cat_features_idx))
])

print("Entrenando Regresión Lineal...")
pipeline_lr.fit(X_train, y_train)

print("Entrenando Random Forest...")
pipeline_rf.fit(X_train, y_train)

print("Entrenando CatBoost...")
pipeline_cb.fit(X_train, y_train)

print("Modelos entrenados correctamente")

y_pred_lr = pipeline_lr.predict(X_test)
y_pred_rf = pipeline_rf.predict(X_test)
y_pred_cb_log = pipeline_cb.predict(X_test)
y_pred_cb = np.expm1(y_pred_cb_log)
y_real = np.expm1(y_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_cb = mean_squared_error(y_real, y_pred_cb)

rmse_lr = np.sqrt(mse_lr)
rmse_rf = np.sqrt(mse_rf)
rmse_cb = np.sqrt(mse_cb)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_cb = mean_absolute_error(y_real, y_pred_cb)

r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
r2_cb = r2_score(y_real, y_pred_cb)

print("\n=== EVALUACIÓN DE MODELOS PREDICTIVOS ===")
print(f"Regresión Lineal - R²: {r2_lr:.4f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.3f}")
print(f"Random Forest     - R²: {r2_rf:.4f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.3f}")
print(f"CatBoost          - R²: {r2_cb:.4f}, RMSE: {rmse_cb:.2f}, MAE: {mae_cb:.3f}")



# Visualización Regresión Lineal
residuals_lr = np.expm1(y_test) - np.expm1(y_pred_lr)

plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred_lr), alpha=0.5)
plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--')
plt.xlabel('Valor Real')
plt.ylabel('Predicción Regresión Lineal')
plt.title('Regresión Lineal: Predicción vs Valor Real')
plt.grid(True)
plt.tight_layout()
plt.savefig('lr_pred_vs_real.png')
print("Gráfico guardado: lr_pred_vs_real.png")

plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_pred_lr), residuals_lr, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicción Regresión Lineal')
plt.ylabel('Residuo')
plt.title('Regresión Lineal: Análisis de Residuos')
plt.grid(True)
plt.tight_layout()
plt.savefig('lr_residuos.png')
print("Gráfico guardado: lr_residuos.png")

plt.figure(figsize=(10, 6))
sns.histplot(residuals_lr, kde=True, bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Regresión Lineal: Distribución de Errores')
plt.tight_layout()
plt.savefig('lr_distribucion_errores.png')
print("Gráfico guardado: lr_distribucion_errores.png")

# Visualización Random Forest
residuals_rf = np.expm1(y_test) - np.expm1(y_pred_rf)

plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_test), np.expm1(y_pred_rf), alpha=0.5)
plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], [np.expm1(y_test).min(), np.expm1(y_test).max()], 'r--')
plt.xlabel('Valor Real')
plt.ylabel('Predicción Random Forest')
plt.title('Random Forest: Predicción vs Valor Real')
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_pred_vs_real.png')
print("Gráfico guardado: rf_pred_vs_real.png")

plt.figure(figsize=(10, 6))
plt.scatter(np.expm1(y_pred_rf), residuals_rf, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicción Random Forest')
plt.ylabel('Residuo')
plt.title('Random Forest: Análisis de Residuos')
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_residuos.png')
print("Gráfico guardado: rf_residuos.png")

plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, kde=True, bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Random Forest: Distribución de Errores')
plt.tight_layout()
plt.savefig('rf_distribucion_errores.png')
print("Gráfico guardado: rf_distribucion_errores.png")

# Visualización CatBoost
residuals_cb = y_real - y_pred_cb

plt.figure(figsize=(10, 6))
plt.scatter(y_real, y_pred_cb, alpha=0.5)
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--')
plt.xlabel('Valor Real')
plt.ylabel('Predicción CatBoost')
plt.title('CatBoost: Predicción vs Valor Real')
plt.grid(True)
plt.tight_layout()
plt.savefig('cb_pred_vs_real.png')
print("Gráfico guardado: cb_pred_vs_real.png")

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_cb, residuals_cb, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicción CatBoost')
plt.ylabel('Residuo')
plt.title('CatBoost: Análisis de Residuos')
plt.grid(True)
plt.tight_layout()
plt.savefig('cb_residuos.png')
print("Gráfico guardado: cb_residuos.png")

plt.figure(figsize=(10, 6))
sns.histplot(residuals_cb, kde=True, bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('CatBoost: Distribución de Errores')
plt.tight_layout()
plt.savefig('cb_distribucion_errores.png')
print("Gráfico guardado: cb_distribucion_errores.png")

# (omitido por espacio, ya está incluido en el documento actual de Canvas)
import os
os.makedirs("resultados_md", exist_ok=True)

modelos = {
    "Regresión Lineal": {
        "r2": r2_lr,
        "rmse": rmse_lr,
        "mae": mae_lr,
        "pred": np.expm1(y_pred_lr),
        "resid": residuals_lr,
        "file": "resultados_md/prediccion_lr.md"
    },
    "Random Forest": {
        "r2": r2_rf,
        "rmse": rmse_rf,
        "mae": mae_rf,
        "pred": np.expm1(y_pred_rf),
        "resid": residuals_rf,
        "file": "resultados_md/prediccion_rf.md"
    },
    "CatBoost": {
        "r2": r2_cb,
        "rmse": rmse_cb,
        "mae": mae_cb,
        "pred": y_pred_cb,
        "resid": residuals_cb,
        "file": "resultados_md/prediccion_cb.md"
    }
}

for nombre, data in modelos.items():
    with open(data['file'], 'w') as f:
        f.write(f"# Resultados de Predicción: {nombre}\n\n")
        f.write("## Resumen de Métricas\n\n")
        f.write(f"- **R²**: {data['r2']:.4f}\n")
        f.write(f"- **RMSE**: {data['rmse']:.2f}\n")
        f.write(f"- **MAE**: {data['mae']:.2f}\n\n")

        f.write("## Interpretación\n\n")
        f.write(f"Este modelo explica aproximadamente el {data['r2']*100:.1f}% de la variabilidad en las mermas.\n")
        f.write(f"En promedio, las predicciones difieren de los valores reales en ±{data['rmse']:.2f} unidades.\n\n")

        f.write("## Muestra de Predicciones (Top 10)\n\n")
        f.write("| # | Valor Real | Predicción | Error |\n")
        f.write("|---|------------|------------|--------|\n")
        for i in range(10):
            real = y_real.iloc[i]
            pred = data['pred'][i]
            err = real - pred
            f.write(f"| {i+1} | {real:.2f} | {pred:.2f} | {err:.2f} |\n")

        f.write("\n## Distribución del Error\n\n")
        f.write(f"- **Error Mínimo**: {data['resid'].min():.2f}\n")
        f.write(f"- **Error Máximo**: {data['resid'].max():.2f}\n")
        f.write(f"- **Error Promedio**: {data['resid'].mean():.2f}\n")
        f.write(f"- **Desviación Estándar del Error**: {data['resid'].std():.2f}\n")
        f.write("\n*Nota: Un error negativo indica sobreestimación; positivo, subestimación.*\n")

print("Archivos Markdown generados en la carpeta 'resultados_md'.")

model_cb = pipeline_cb.named_steps['regressor'].model 

importances = model_cb.get_feature_importance()
feature_names = np.array(X_train.columns)  
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 8))
plt.barh(feature_names[indices], importances[indices])
plt.xlabel('Importancia')
plt.title('Importancia de Características - CatBoost')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('cb_importancia_caracteristicas.png')