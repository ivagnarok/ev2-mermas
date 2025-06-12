import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
archivo = "mermas_actividad_unidad_2.xlsx"  # Cambia según tu ruta

# === CARGA DEL ARCHIVO ===
df = pd.read_excel(archivo)

# === AGREGAR VARIABLES TEMPORALES DERIVADAS ===
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=True)
df['semana'] = df['fecha'].dt.isocalendar().week.astype(int)
df['dia_semana'] = df['fecha'].dt.dayofweek
df['es_finde'] = df['dia_semana'] >= 5

import holidays
cl_holidays = holidays.country_holidays('CL', years=df['fecha'].dt.year.unique())
df['es_feriado'] = df['fecha'].isin(cl_holidays)

# === FILTRAR COLUMNAS: omitir las que contienen "merma" excepto "merma_unidad_p"
columnas_validas = [col for col in df.columns if 'merma' not in col.lower() or col.lower() == 'merma_unidad_p']
df_filtrado = df[columnas_validas].copy()

# === CONVERSIÓN EXPLÍCITA DE 'mes' y 'año' ===
meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'setiembre': 9, 'octubre': 10,
    'noviembre': 11, 'diciembre': 12
}
if 'mes' in df_filtrado.columns:
    df_filtrado['mes'] = df_filtrado['mes'].astype(str).str.lower().map(meses)

if 'año' in df_filtrado.columns:
    df_filtrado['año'] = pd.to_numeric(df_filtrado['año'], errors='coerce')

# === CONVERTIR VARIABLES CATEGÓRICAS A NUMÉRICAS
for col in df_filtrado.select_dtypes(include=['object', 'category']).columns:
    df_filtrado[col] = df_filtrado[col].astype('category').cat.codes

# === ELIMINAR VARIABLES CONSTANTES
df_filtrado = df_filtrado.loc[:, df_filtrado.nunique() > 1]

# === ELIMINAR FILAS CON NAN
df_filtrado = df_filtrado.dropna()

# === VERIFICAR VARIABLE OBJETIVO
if 'merma_unidad_p' not in df_filtrado.columns:
    raise ValueError("La columna 'merma_unidad_p' no está presente o no es válida.")

# === CALCULAR CORRELACIÓN (SPEARMAN)
correlaciones = df_filtrado.corr(method='spearman')['merma_unidad_p'].drop('merma_unidad_p')
correlaciones_ordenadas = correlaciones.sort_values(ascending=False)

# === FILTRAR POR UMBRAL MÍNIMO DE CORRELACIÓN
umbral = 0.1
correlaciones_significativas = correlaciones_ordenadas[abs(correlaciones_ordenadas) > umbral]

# === MOSTRAR RESULTADOS
print("\nCorrelación con 'merma_unidad_p' (método Spearman, umbral ±0.1):\n")
for var, corr in correlaciones_significativas.items():
    tipo = "positiva" if corr > 0 else "negativa"
    print(f"{var}: {corr:.4f} ({tipo})")

# === VISUALIZACIÓN DE HEATMAP (TOP POSITIVAS Y NEGATIVAS)
top_pos = correlaciones_ordenadas.head(5).index.tolist()
top_neg = correlaciones_ordenadas.tail(5).index.tolist()
top_vars = top_pos + top_neg + ['merma_unidad_p']

df_corr_top = df_filtrado[top_vars].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr_top, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de correlación - Top variables correlacionadas con merma_unidad_p')
plt.tight_layout()
plt.show()