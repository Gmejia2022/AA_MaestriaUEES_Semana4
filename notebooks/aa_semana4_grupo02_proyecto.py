# -*- coding: utf-8 -*-
"""
AA_Semana4_Grupo02_Proyecto.py
==============================
Notebook integrado - Pipeline completo de Machine Learning

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan
  - Ingeniero David Perugachi Rojas

Docente: Ing. Gladys Maria Villegas Rugel
Fecha: 23 Febrero 2026

Objetivos:
  - Implementar un modelo de ML aplicado a un problema real considerando aspectos
    de calidad de datos y mitigacion de sesgos.
  - Aplicar tecnicas de explicabilidad (XAI) para mejorar la transparencia y la
    comprension de las decisiones del modelo.
  - Reflexionar sobre los principios eticos en el diseno e implementacion de
    sistemas automatizados.
  - Documentar correctamente el flujo completo del proyecto incluyendo evaluacion,
    justificacion y comunicacion de resultados.

Descripcion:
  Este notebook integra de forma secuencial los 5 scripts de la carpeta scr/:
    1_ExploracionEDA.py       -> Analisis Exploratorio de Datos
    2_PreProcesamiento.py     -> Preprocesamiento y division de datos
    3_EntrenarYEvaluar.py     -> Entrenamiento de 3 clasificadores
    4_ComparacionExperimental.py -> Comparacion experimental
    5_ProbarModelosProduccion.py -> Prueba en produccion con JSON

  Dataset: Data/DataSet2024.csv
  Resultados: results/ (imagenes PNG)
  Modelos exportados: Models/ (archivos .pkl)
"""

# =============================================================
# IMPORTACIONES GLOBALES
# =============================================================
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
import joblib

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.max_open_warning": 0})

# =============================================================
# CONFIGURACION DE RUTAS
# =============================================================
# BASE_DIR apunta a la raiz del repositorio desde notebooks/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
JSON_PATH = os.path.join(BASE_DIR, "Data", "datos_prueba_produccion.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Guarda una figura en la carpeta results."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =============================================================
# SECCION 1: ANALISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================
# Equivalente a: scr/1_ExploracionEDA.py
# =============================================================
print("=" * 60)
print("  SECCION 1: ANALISIS EXPLORATORIO DE DATOS (EDA)")
print("  Dataset: Empresas del Ecuador - 2024")
print("=" * 60)

# --- 1.1 Carga del dataset ---
df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")

# Normalizar nombres de columnas
df.columns = (
    df.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)

print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nColumnas: {list(df.columns)}")

# --- 1.2 Descripcion de variables ---
print("\n" + "=" * 60)
print("  1.2 DESCRIPCION DE VARIABLES")
print("=" * 60)

print("\n--- Tipos de datos ---")
print(df.dtypes)

print("\n--- Valores nulos por columna ---")
nulos = df.isnull().sum()
print(nulos[nulos > 0] if nulos.sum() > 0 else "No hay valores nulos.")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\nVariables numericas ({len(num_cols)}): {num_cols}")
print(f"Variables categoricas ({len(cat_cols)}): {cat_cols}")

print("\n--- Estadisticas descriptivas (variables numericas) ---")
desc = df[num_cols].describe().T
desc["coef_var"] = (desc["std"] / desc["mean"]).abs()
print(desc.to_string())

print("\n--- Distribucion de variables categoricas ---")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10).to_string())

# --- 1.3 Limpieza basica para visualizaciones ---
for col in df.columns:
    if col not in cat_cols and col != "Ano":
        df[col] = pd.to_numeric(df[col], errors="coerce")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
fin_cols = [c for c in num_cols if c != "Ano"]

# --- 1.4 Creacion de variable objetivo (Desempeno) ---
print("\n" + "=" * 60)
print("  1.4 CREACION DE VARIABLE OBJETIVO")
print("=" * 60)

epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df["ROA"] = df["UtilidadNeta"] / (df["Activo"] + epsilon)
df["ROE"] = df["UtilidadNeta"] / (df["Patrimonio"] + epsilon)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Margen_Neto"])

df["Desempeno"] = pd.qcut(
    df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop"
)

print(f"\nDistribucion de la variable objetivo (Desempeno):")
print(df["Desempeno"].value_counts())
print(f"\nPorcentaje:")
print((df["Desempeno"].value_counts(normalize=True) * 100).round(2))

# --- 1.5 Visualizaciones del EDA ---
print("\n" + "=" * 60)
print("  1.5 GENERACION DE VISUALIZACIONES EDA")
print("=" * 60)

palette_desemp = {"Bajo": "#e74c3c", "Medio": "#f39c12", "Alto": "#27ae60"}
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# [1/8] Distribucion de la variable objetivo
print("\n[1/8] Distribucion de la variable objetivo...")
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["Desempeno"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[palette_desemp[x] for x in counts.index],
              edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title("Distribucion de la Variable Objetivo: Desempeno Financiero",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Nivel de Desempeno")
ax.set_ylabel("Cantidad de Empresas")
save_fig(fig, "01_distribucion_variable_objetivo.png")

# [2/8] Distribucion por Sector
print("[2/8] Distribucion por Sector...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sector_counts = df["Sector"].value_counts()
axes[0].bar(sector_counts.index, sector_counts.values,
            color=["#3498db", "#e67e22"], edgecolor="black", linewidth=0.5)
for i, (idx, val) in enumerate(sector_counts.items()):
    axes[0].text(i, val + 500, f"{val:,}", ha="center", fontweight="bold")
axes[0].set_title("Cantidad de Empresas por Sector", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad")
ct = pd.crosstab(df["Sector"], df["Desempeno"], normalize="index") * 100
ct[["Bajo", "Medio", "Alto"]].plot(
    kind="bar", stacked=True, ax=axes[1],
    color=[palette_desemp["Bajo"], palette_desemp["Medio"], palette_desemp["Alto"]],
    edgecolor="black", linewidth=0.5)
axes[1].set_title("Desempeno Financiero por Sector (%)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_xlabel("Sector")
axes[1].legend(title="Desempeno")
axes[1].tick_params(axis="x", rotation=0)
fig.tight_layout()
save_fig(fig, "02_distribucion_sector.png")

# [3/8] Histogramas de variables financieras
print("[3/8] Histogramas de variables financieras...")
plot_cols = ["Cant_Empleados", "Activo", "Patrimonio", "IngresoVentas",
             "UtilidadAntesImpuestos", "UtilidadEjercicio", "UtilidadNeta",
             "IR_Causado", "IngresosTotales"]
plot_cols = [c for c in plot_cols if c in df.columns]
n = len(plot_cols)
ncols = 3
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    data = df[col].dropna()
    if data.max() > 1e6:
        log_data = np.log1p(data.clip(lower=0))
        axes[i].hist(log_data, bins=50, color="#3498db", edgecolor="black",
                     linewidth=0.3, alpha=0.8)
        axes[i].set_xlabel(f"log(1 + {col})")
    else:
        axes[i].hist(data, bins=50, color="#3498db", edgecolor="black",
                     linewidth=0.3, alpha=0.8)
        axes[i].set_xlabel(col)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Frecuencia")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Distribucion de Variables Financieras", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "03_histogramas_variables_financieras.png")

# [4/8] Boxplots por nivel de desempeno
print("[4/8] Boxplots por nivel de desempeno...")
indicadores = ["Margen_Neto", "ROA", "ROE"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(indicadores):
    data = df[[col, "Desempeno"]].dropna()
    q01 = data[col].quantile(0.01)
    q99 = data[col].quantile(0.99)
    data_clip = data[(data[col] >= q01) & (data[col] <= q99)]
    sns.boxplot(data=data_clip, x="Desempeno", y=col, order=["Bajo", "Medio", "Alto"],
                palette=palette_desemp, ax=axes[i], fliersize=2)
    axes[i].set_title(f"{col} por Nivel de Desempeno", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Desempeno")
    axes[i].set_ylabel(col)
fig.suptitle("Boxplots de Indicadores Financieros por Desempeno",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "04_boxplots_indicadores_desempeno.png")

# [5/8] Boxplots de variables financieras originales
print("[5/8] Boxplots de variables financieras originales...")
box_cols = ["Activo", "Patrimonio", "IngresoVentas", "UtilidadNeta", "IngresosTotales"]
box_cols = [c for c in box_cols if c in df.columns]
fig, axes = plt.subplots(1, len(box_cols), figsize=(4 * len(box_cols), 6))
if len(box_cols) == 1:
    axes = [axes]
for i, col in enumerate(box_cols):
    data = df[[col, "Desempeno"]].dropna().copy()
    data["log_val"] = np.log1p(data[col].clip(lower=0))
    sns.boxplot(data=data, x="Desempeno", y="log_val", order=["Bajo", "Medio", "Alto"],
                palette=palette_desemp, ax=axes[i], fliersize=1)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel(f"log(1 + {col})")
    axes[i].set_xlabel("Desempeno")
fig.suptitle("Variables Financieras por Nivel de Desempeno (escala log)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "05_boxplots_variables_financieras.png")

# [6/8] Matriz de correlacion
print("[6/8] Matriz de correlacion...")
corr_cols = [c for c in fin_cols if c in df.columns] + ["Margen_Neto", "ROA", "ROE"]
corr_cols = list(dict.fromkeys(corr_cols))
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Matriz de Correlacion - Variables Financieras e Indicadores",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save_fig(fig, "06_matriz_correlacion.png")

# [7/8] Pairplot de indicadores clave
print("[7/8] Pairplot de indicadores clave (muestra)...")
sample_size = min(3000, len(df))
df_sample = df[["Margen_Neto", "ROA", "ROE", "Desempeno"]].dropna().sample(
    n=sample_size, random_state=42)
for col in ["Margen_Neto", "ROA", "ROE"]:
    q01 = df_sample[col].quantile(0.01)
    q99 = df_sample[col].quantile(0.99)
    df_sample = df_sample[(df_sample[col] >= q01) & (df_sample[col] <= q99)]
g = sns.pairplot(df_sample, hue="Desempeno", palette=palette_desemp,
                 hue_order=["Bajo", "Medio", "Alto"],
                 diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
g.figure.suptitle("Relaciones entre Indicadores Financieros por Desempeno",
                  fontsize=14, fontweight="bold", y=1.02)
save_fig(g.figure, "07_pairplot_indicadores.png")

# [8/8] Tabla estadisticas por clase
print("[8/8] Tabla de estadisticas por clase de desempeno...")
stats_cols = ["Cant_Empleados", "Activo", "Patrimonio", "IngresoVentas",
              "UtilidadNeta", "IngresosTotales", "Margen_Neto", "ROA", "ROE"]
stats_cols = [c for c in stats_cols if c in df.columns]
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis("off")
summary_data = []
for nivel in ["Bajo", "Medio", "Alto"]:
    row = [nivel]
    for col in stats_cols:
        mean_val = df[df["Desempeno"] == nivel][col].mean()
        row.append(f"{mean_val:,.0f}" if abs(mean_val) > 1000 else f"{mean_val:.4f}")
    summary_data.append(row)
table = ax.table(cellText=summary_data, colLabels=["Desempeno"] + stats_cols,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.8)
for j in range(len(stats_cols) + 1):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")
row_colors = {"Bajo": "#fadbd8", "Medio": "#fdebd0", "Alto": "#d5f5e3"}
for i, nivel in enumerate(["Bajo", "Medio", "Alto"]):
    for j in range(len(stats_cols) + 1):
        table[i + 1, j].set_facecolor(row_colors[nivel])
ax.set_title("Estadisticas Descriptivas por Nivel de Desempeno (Media)",
             fontsize=14, fontweight="bold", pad=20)
save_fig(fig, "08_tabla_estadisticas_por_clase.png")

print("\n" + "=" * 60)
print("  RESUMEN DEL EDA")
print("=" * 60)
print(f"  Total de registros analizados: {len(df):,}")
print(f"  Sectores: {df['Sector'].nunique()}")
print(f"  Clases de desempeno: {df['Desempeno'].nunique()}")
print(f"  Graficos generados: 8 (01 - 08)")
print("=" * 60)


# =============================================================
# SECCION 2: PREPROCESAMIENTO DE DATOS
# =============================================================
# Equivalente a: scr/2_PreProcesamiento.py
# =============================================================
print("\n" + "=" * 60)
print("  SECCION 2: PREPROCESAMIENTO DE DATOS")
print("=" * 60)

# Recargar dataset limpio
df2 = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
df2.columns = (
    df2.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)
print(f"\nDataset original: {df2.shape[0]} filas x {df2.shape[1]} columnas")

# --- 2.1 Eliminar columna Ano ---
print("\n--- 2.1 Eliminacion de columna Ano ---")
col_ano = [c for c in df2.columns if df2[c].nunique() == 1 and df2[c].dtype in ["int64", "float64"]]
if not col_ano:
    col_ano = [df2.columns[0]]
print(f"Columna identificada: '{col_ano[0]}' (valor unico: {df2[col_ano[0]].unique()})")
df2 = df2.drop(columns=col_ano)
print(f"Dataset tras eliminar Ano: {df2.shape[0]} filas x {df2.shape[1]} columnas")

# --- 2.2 Tratamiento de valores nulos ---
print("\n--- 2.2 Tratamiento de valores nulos ---")
cat_cols2 = df2.select_dtypes(include=["object"]).columns.tolist()
for col in df2.columns:
    if col not in cat_cols2:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")
num_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
nulos_antes = df2.isnull().sum()
total_nulos = nulos_antes.sum()
print(f"Total de valores nulos: {total_nulos}")
if total_nulos > 0:
    for col in num_cols2:
        if df2[col].isnull().sum() > 0:
            mediana = df2[col].median()
            df2[col] = df2[col].fillna(mediana)
    for col in cat_cols2:
        if df2[col].isnull().sum() > 0:
            df2[col] = df2[col].fillna(df2[col].mode()[0])
print(f"Nulos finales: {df2.isnull().sum().sum()}")

# --- 2.3 Creacion de variable objetivo ---
print("\n--- 2.3 Creacion de variable objetivo ---")
df2["Margen_Neto"] = df2["UtilidadNeta"] / (df2["IngresosTotales"] + 1e-7)
df2 = df2.replace([np.inf, -np.inf], np.nan)
df2 = df2.dropna(subset=["Margen_Neto"])
df2["Desempeno"] = pd.qcut(
    df2["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop"
)
print(f"Registros con variable objetivo: {len(df2):,}")
print(f"\nDistribucion:")
print(df2["Desempeno"].value_counts())

# --- 2.4 Codificacion de variables categoricas ---
print("\n--- 2.4 Codificacion de variables categoricas ---")
cat_to_encode = df2.select_dtypes(include=["object", "category"]).columns.tolist()
cat_to_encode = [c for c in cat_to_encode if c != "Desempeno"]
print(f"Variables categoricas a codificar: {cat_to_encode}")

encoders = {}
for col in cat_to_encode:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

le_target = LabelEncoder()
df2["Desempeno_cod"] = le_target.fit_transform(df2["Desempeno"])
print(f"\n  Desempeno: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# --- 2.5 Escalado de variables numericas ---
print("\n--- 2.5 Escalado de variables numericas (StandardScaler) ---")
exclude = ["Desempeno", "Desempeno_cod", "Margen_Neto"]
feature_cols = [c for c in df2.columns if c not in exclude]
print(f"Features para el modelo ({len(feature_cols)}): {feature_cols}")

X = df2[feature_cols].copy()
y = df2["Desempeno_cod"].copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

# --- 2.6 Division entrenamiento / prueba (80/20) ---
print("\n--- 2.6 Division entrenamiento / prueba (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Dataset total:       {X_scaled.shape[0]:,} registros")
print(f"  Entrenamiento (80%): {X_train.shape[0]:,} registros")
print(f"  Prueba (20%):        {X_test.shape[0]:,} registros")
print(f"  Features:            {X_train.shape[1]}")

dist_train = y_train.value_counts(normalize=True).sort_index() * 100
dist_test = y_test.value_counts(normalize=True).sort_index() * 100
dist_df = pd.DataFrame({"Entrenamiento (%)": dist_train.round(2), "Prueba (%)": dist_test.round(2)})
dist_df.index = [le_target.inverse_transform([i])[0] for i in dist_df.index]
print(f"\n  Distribucion de clases (estratificada):")
print(dist_df.to_string())

# --- 2.7 Visualizaciones del preprocesamiento ---
print("\n--- 2.7 Visualizaciones del preprocesamiento ---")
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# [1/4] Comparativa antes/despues del escalado
print("\n[1/4] Comparativa antes/despues del escalado...")
sample_cols = feature_cols[:6]
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
X[sample_cols].boxplot(ax=axes[0], patch_artist=True,
                        boxprops=dict(facecolor="#3498db", alpha=0.6),
                        medianprops=dict(color="red", linewidth=2))
axes[0].set_title("ANTES del Escalado (valores originales)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Valor")
axes[0].tick_params(axis="x", rotation=30)
X_scaled[sample_cols].boxplot(ax=axes[1], patch_artist=True,
                               boxprops=dict(facecolor="#27ae60", alpha=0.6),
                               medianprops=dict(color="red", linewidth=2))
axes[1].set_title("DESPUES del Escalado (StandardScaler)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Valor escalado")
axes[1].tick_params(axis="x", rotation=30)
fig.suptitle("Efecto del Escalado en Variables Numericas", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "09_comparativa_escalado.png")

# [2/4] Distribucion train/test por clase
print("[2/4] Distribucion train/test por clase...")
clases_pp = le_target.classes_
colors_pp = [palette_desemp[c] for c in clases_pp]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
train_counts = y_train.value_counts().sort_index()
train_labels = [le_target.inverse_transform([i])[0] for i in train_counts.index]
bars = axes[0].bar(train_labels, train_counts.values, color=colors_pp,
                   edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, train_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=10)
axes[0].set_title(f"Entrenamiento ({X_train.shape[0]:,} registros)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Cantidad")
axes[0].set_xlabel("Desempeno")
test_counts = y_test.value_counts().sort_index()
test_labels = [le_target.inverse_transform([i])[0] for i in test_counts.index]
bars = axes[1].bar(test_labels, test_counts.values, color=colors_pp,
                   edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, test_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=10)
axes[1].set_title(f"Prueba ({X_test.shape[0]:,} registros)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Cantidad")
axes[1].set_xlabel("Desempeno")
fig.suptitle("Division Estratificada: Entrenamiento (80%) vs Prueba (20%)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "10_distribucion_train_test.png")

# [3/4] Distribucion de features escaladas
print("[3/4] Distribucion de features escaladas...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(feature_cols[:10]):
    axes[i].hist(X_scaled[col], bins=50, color="#3498db", edgecolor="black",
                 linewidth=0.3, alpha=0.8)
    axes[i].axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Frecuencia")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Distribucion de Features tras StandardScaler", fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "11_distribucion_features_escaladas.png")

# [4/4] Tabla resumen del preprocesamiento
print("[4/4] Tabla resumen del preprocesamiento...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
resumen_data = [
    ["Registros originales", f"{df2.shape[0] + X_train.shape[0] + X_test.shape[0]:,}"],
    ["Columna eliminada", "Ano (valor unico 2024, sin poder predictivo)"],
    ["Valores nulos", f"{total_nulos} (tratados con mediana/moda)"],
    ["Variable categorica codificada", f"Sector -> LabelEncoder ({len(encoders)} variable)"],
    ["Variable objetivo", "Desempeno (Alto=0, Bajo=1, Medio=2)"],
    ["Metodo de escalado", "StandardScaler (media=0, std=1)"],
    ["Features finales", f"{len(feature_cols)} variables"],
    ["Division", "80% entrenamiento / 20% prueba (estratificada)"],
    ["Registros entrenamiento", f"{X_train.shape[0]:,}"],
    ["Registros prueba", f"{X_test.shape[0]:,}"],
]
table = ax.table(cellText=resumen_data, colLabels=["Concepto", "Detalle"],
                 cellLoc="left", loc="center", colWidths=[0.35, 0.65])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for j in range(2):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(resumen_data) + 1):
    color = "#eaf2f8" if i % 2 == 0 else "#ffffff"
    for j in range(2):
        table[i, j].set_facecolor(color)
ax.set_title("Resumen del Preprocesamiento de Datos", fontsize=14, fontweight="bold", pad=20)
save_fig(fig, "12_resumen_preprocesamiento.png")

print("\n  Preprocesamiento completado exitosamente.")
print(f"  Graficos generados: 4 (09 - 12)")


# =============================================================
# SECCION 3: ENTRENAMIENTO Y EVALUACION DE CLASIFICADORES
# =============================================================
# Equivalente a: scr/3_EntrenarYEvaluar.py
# =============================================================
print("\n" + "=" * 60)
print("  SECCION 3: IMPLEMENTACION DE CLASIFICADORES")
print("  Modelos: Arbol de Decision | SVM | Random Forest")
print("=" * 60)

clases = list(le_target.classes_)

# --- 3.1 Modelo 1: Arbol de Decision ---
print("\n--- 3.1 MODELO 1: ARBOL DE DECISION ---")
t0 = time.time()
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_time = time.time() - t0
y_pred_dt = dt_model.predict(X_test)
print(f"  Tiempo de entrenamiento: {dt_time:.2f}s")
print(f"  Profundidad del arbol:   {dt_model.get_depth()}")
print(f"  Hojas:                   {dt_model.get_n_leaves()}")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_dt, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_dt, target_names=clases))

# --- 3.2 Modelo 2: SVM con ajuste de hiperparametros ---
print("--- 3.2 MODELO 2: SVM (Support Vector Machine) ---")
print("  Ajuste de hiperparametros: kernel y C")
sample_size_svm = min(15000, len(X_train))
idx_sample = np.random.RandomState(42).choice(X_train.index, size=sample_size_svm, replace=False)
X_train_sample = X_train.loc[idx_sample]
y_train_sample = y_train.loc[idx_sample]
print(f"\n  Muestra para GridSearchCV: {sample_size_svm:,} registros")

param_grid_svm = {"kernel": ["rbf", "linear"], "C": [0.1, 1.0, 10.0]}
svm_grid = GridSearchCV(
    SVC(class_weight="balanced", random_state=42),
    param_grid_svm, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=0
)
t0 = time.time()
svm_grid.fit(X_train_sample, y_train_sample)
svm_search_time = time.time() - t0
print(f"  GridSearch completado en {svm_search_time:.2f}s")
print(f"  Mejores hiperparametros: {svm_grid.best_params_}")
print(f"  Mejor F1 (CV): {svm_grid.best_score_:.4f}")

gs_results = pd.DataFrame(svm_grid.cv_results_)
gs_summary = gs_results[["param_kernel", "param_C", "mean_test_score",
                           "std_test_score", "rank_test_score"]].sort_values("rank_test_score")
print(f"\n  Resultados del GridSearch:")
print(gs_summary.to_string(index=False))

t0 = time.time()
svm_model = SVC(
    kernel=svm_grid.best_params_["kernel"],
    C=svm_grid.best_params_["C"],
    class_weight="balanced", random_state=42, probability=True
)
svm_model.fit(X_train_sample, y_train_sample)
svm_time = time.time() - t0
y_pred_svm = svm_model.predict(X_test)
print(f"\n  Tiempo de entrenamiento (modelo final): {svm_time:.2f}s")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_svm, target_names=clases))

# --- 3.3 Modelo 3: Random Forest ---
print("--- 3.3 MODELO 3: RANDOM FOREST ---")
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t0
y_pred_rf = rf_model.predict(X_test)
print(f"  Tiempo de entrenamiento: {rf_time:.2f}s")
print(f"  Arboles:    {rf_model.n_estimators}")
print(f"  Max depth:  {rf_model.max_depth}")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  F1 (weighted): {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"\n  Reporte de Clasificacion:")
print(classification_report(y_test, y_pred_rf, target_names=clases))

# --- 3.4 Comparativa de los 3 modelos ---
print("--- 3.4 COMPARATIVA DE MODELOS ---")
modelos_eval = {
    "Arbol de Decision": {"modelo": dt_model, "pred": y_pred_dt, "tiempo": dt_time},
    "SVM": {"modelo": svm_model, "pred": y_pred_svm, "tiempo": svm_time},
    "Random Forest": {"modelo": rf_model, "pred": y_pred_rf, "tiempo": rf_time},
}
comparativa = []
for nombre, info in modelos_eval.items():
    pred = info["pred"]
    comparativa.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "F1 (weighted)": f1_score(y_test, pred, average="weighted"),
        "Precision (weighted)": precision_score(y_test, pred, average="weighted"),
        "Recall (weighted)": recall_score(y_test, pred, average="weighted"),
        "Tiempo (s)": info["tiempo"],
    })
df_comp = pd.DataFrame(comparativa)
print(f"\n{df_comp.to_string(index=False)}")
mejor = df_comp.loc[df_comp["F1 (weighted)"].idxmax()]
print(f"\n  MEJOR MODELO: {mejor['Modelo']} (F1 = {mejor['F1 (weighted)']:.4f})")

# --- 3.5 Visualizaciones de entrenamiento ---
print("\n--- 3.5 VISUALIZACIONES ---")
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# [1/5] Matrices de confusion comparativa
print("\n[1/5] Matrices de confusion...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, (nombre, info) in enumerate(modelos_eval.items()):
    cm = confusion_matrix(y_test, info["pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=axes[i],
                vmin=0, vmax=1, linewidths=0.5)
    acc = accuracy_score(y_test, info["pred"])
    f1 = f1_score(y_test, info["pred"], average="weighted")
    axes[i].set_title(f"{nombre}\nAcc={acc:.3f} | F1={f1:.3f}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Prediccion")
    axes[i].set_ylabel("Real")
fig.suptitle("Matrices de Confusion Normalizadas - Comparativa de Modelos",
             fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, "13_matrices_confusion_comparativa.png")

# [2/5] Comparativa de metricas
print("[2/5] Comparativa de metricas...")
fig, ax = plt.subplots(figsize=(12, 6))
metricas_plot = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"]
x = np.arange(len(metricas_plot))
width = 0.25
colors_m = ["#3498db", "#e67e22", "#27ae60"]
for i, (_, row) in enumerate(df_comp.iterrows()):
    valores = [row[m] for m in metricas_plot]
    bars = ax.bar(x + i * width, valores, width, label=row["Modelo"],
                  color=colors_m[i], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Valor")
ax.set_title("Comparativa de Metricas entre Modelos", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metricas_plot)
ax.legend(loc="lower right")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "14_comparativa_metricas.png")

# [3/5] Arbol de Decision - visualizacion
print("[3/5] Visualizacion del Arbol de Decision (primeros niveles)...")
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(dt_model, max_depth=3, feature_names=feature_cols, class_names=clases,
          filled=True, rounded=True, fontsize=8, ax=ax, proportion=True, impurity=True)
ax.set_title("Arbol de Decision (primeros 3 niveles)", fontsize=16, fontweight="bold")
save_fig(fig, "15_arbol_decision_visualizacion.png")

# [4/5] Importancia de features (Random Forest)
print("[4/5] Importancia de features (Random Forest)...")
importancias = rf_model.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_cols, "Importancia": importancias}).sort_values(
    "Importancia", ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = plt.cm.viridis(feat_imp["Importancia"] / feat_imp["Importancia"].max())
bars = ax.barh(feat_imp["Feature"], feat_imp["Importancia"], color=colors_bar,
               edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, feat_imp["Importancia"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Importancia de Features - Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importancia (Gini)")
ax.grid(axis="x", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "16_importancia_features_rf.png")

# [5/5] GridSearch SVM
print("[5/5] Resultados del GridSearch SVM...")
fig, ax = plt.subplots(figsize=(10, 6))
for kernel in ["rbf", "linear"]:
    mask = gs_results["param_kernel"] == kernel
    subset = gs_results[mask].sort_values("param_C")
    ax.plot(subset["param_C"].astype(float), subset["mean_test_score"],
            marker="o", linewidth=2, markersize=8, label=f"kernel={kernel}")
    ax.fill_between(subset["param_C"].astype(float),
                    subset["mean_test_score"] - subset["std_test_score"],
                    subset["mean_test_score"] + subset["std_test_score"], alpha=0.15)
best_c = svm_grid.best_params_["C"]
best_k = svm_grid.best_params_["kernel"]
ax.axvline(x=best_c, color="red", linestyle="--", alpha=0.7,
           label=f"Mejor: C={best_c}, {best_k}")
ax.set_xscale("log")
ax.set_xlabel("Parametro C (escala log)", fontsize=12)
ax.set_ylabel("F1-Score (CV)", fontsize=12)
ax.set_title("GridSearchCV - SVM: Ajuste de Kernel y C", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "17_gridsearch_svm.png")

# --- 3.6 Exportar modelos ---
print("\n--- 3.6 EXPORTACION DE MODELOS ---")
le_sector_export = encoders.get("Sector", None)

joblib.dump(dt_model, os.path.join(MODELS_DIR, "arbol_decision.pkl"))
joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm.pkl"))
joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le_target, os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
if le_sector_export:
    joblib.dump(le_sector_export, os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

print(f"  Modelos exportados en: {MODELS_DIR}")
print(f"  Archivos: arbol_decision.pkl, svm.pkl, random_forest.pkl")
print(f"  Artefactos: scaler.pkl, label_encoder_target.pkl, label_encoder_sector.pkl, feature_columns.pkl")
print(f"\n  MEJOR MODELO: {mejor['Modelo']} (F1 = {mejor['F1 (weighted)']:.4f})")
print(f"  Feature mas importante (RF): {feat_imp.iloc[-1]['Feature']} ({feat_imp.iloc[-1]['Importancia']:.4f})")
print("  Graficos generados: 5 (13 - 17)")


# =============================================================
# SECCION 4: COMPARACION EXPERIMENTAL
# =============================================================
# Equivalente a: scr/4_ComparacionExperimental.py
# =============================================================
print("\n" + "=" * 60)
print("  SECCION 4: COMPARACION EXPERIMENTAL DE MODELOS")
print("=" * 60)

# Reutilizamos X_train, X_test, y_train, y_test ya preparados
# Reentrenar con los mismos hiperparametros para reproducibilidad
modelos_config = {
    "Arbol de Decision": DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    ),
    f"SVM ({svm_grid.best_params_['kernel']}, C={svm_grid.best_params_['C']})": SVC(
        kernel=svm_grid.best_params_["kernel"],
        C=svm_grid.best_params_["C"],
        class_weight="balanced", random_state=42, probability=True
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42
    ),
}

resultados_exp = {}
for nombre, modelo in modelos_config.items():
    print(f"\n  Entrenando {nombre}...")
    t0 = time.time()
    if "SVM" in nombre:
        modelo.fit(X_train_sample, y_train_sample)
    else:
        modelo.fit(X_train, y_train)
    t_train = time.time() - t0
    y_pred = modelo.predict(X_test)
    resultados_exp[nombre] = {"modelo": modelo, "y_pred": y_pred, "tiempo": t_train}
    print(f"    Completado en {t_train:.2f}s")

# --- 4.1 Metricas detalladas por clase ---
print("\n--- 4.1 METRICAS DETALLADAS POR MODELO Y POR CLASE ---")
metricas_por_clase = {}
metricas_globales = []

for nombre, info in resultados_exp.items():
    y_pred = info["y_pred"]
    print(f"\n  --- {nombre} ---")
    report = classification_report(y_test, y_pred, target_names=clases, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=clases))
    for clase in clases:
        metricas_por_clase[f"{nombre} | {clase}"] = {
            "Modelo": nombre, "Clase": clase,
            "Precision": report[clase]["precision"],
            "Recall": report[clase]["recall"],
            "F1-Score": report[clase]["f1-score"],
            "Support": report[clase]["support"],
        }
    metricas_globales.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted"),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score (weighted)": f1_score(y_test, y_pred, average="weighted"),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro"),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro"),
        "F1-Score (macro)": f1_score(y_test, y_pred, average="macro"),
        "Tiempo (s)": info["tiempo"],
    })

df_por_clase = pd.DataFrame(metricas_por_clase.values())
df_globales = pd.DataFrame(metricas_globales)

print("\n--- TABLA RESUMEN - METRICAS GLOBALES (weighted) ---")
print(df_globales[["Modelo", "Accuracy", "Precision (weighted)",
                    "Recall (weighted)", "F1-Score (weighted)", "Tiempo (s)"]].to_string(index=False))

mejor_exp_idx = df_globales["F1-Score (weighted)"].idxmax()
mejor_exp = df_globales.loc[mejor_exp_idx]
print(f"\n  MEJOR MODELO: {mejor_exp['Modelo']} (F1 weighted = {mejor_exp['F1-Score (weighted)']:.4f})")

nombres_modelo_exp = list(resultados_exp.keys())
colores_modelo_exp = {
    nombres_modelo_exp[0]: "#3498db",
    nombres_modelo_exp[1]: "#e67e22",
    nombres_modelo_exp[2]: "#27ae60",
}

# --- 4.2 Visualizaciones de comparacion ---
print("\n--- 4.2 VISUALIZACIONES ---")

# [1/6] Matrices de confusion detalladas
print("\n[1/6] Matrices de confusion (absolutas y normalizadas)...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for i, (nombre, info) in enumerate(resultados_exp.items()):
    cm = confusion_matrix(y_test, info["y_pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=axes[0, i],
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[0, i].set_title(f"{nombre}\n(Valores Absolutos)", fontsize=12, fontweight="bold")
    axes[0, i].set_xlabel("Prediccion")
    axes[0, i].set_ylabel("Real")
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="YlOrRd",
                xticklabels=clases, yticklabels=clases, ax=axes[1, i],
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1, i].set_title(f"{nombre}\n(Normalizadas por clase real)", fontsize=12, fontweight="bold")
    axes[1, i].set_xlabel("Prediccion")
    axes[1, i].set_ylabel("Real")
fig.suptitle("Matrices de Confusion - Comparacion Experimental",
             fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "18_matrices_confusion_detalladas.png")

# [2/6] Barplot de metricas globales
print("[2/6] Barplot de metricas globales...")
fig, ax = plt.subplots(figsize=(14, 7))
metricas_plot2 = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-Score (weighted)"]
x = np.arange(len(metricas_plot2))
width = 0.25
for i, nombre in enumerate(nombres_modelo_exp):
    row = df_globales[df_globales["Modelo"] == nombre].iloc[0]
    valores = [row[m] for m in metricas_plot2]
    bars = ax.bar(x + i * width, valores, width, label=nombre,
                  color=colores_modelo_exp[nombre], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Valor de la Metrica", fontsize=12)
ax.set_title("Comparacion de Metricas Globales (weighted) por Modelo", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace(" (weighted)", "\n(weighted)") for m in metricas_plot2], fontsize=10)
ax.legend(fontsize=11, loc="lower right")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "19_barplot_metricas_globales.png")

# [3/6] Heatmap de metricas por clase
print("[3/6] Heatmap de metricas por clase...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, metrica in enumerate(["Precision", "Recall", "F1-Score"]):
    pivot = df_por_clase.pivot(index="Modelo", columns="Clase", values=metrica)
    pivot = pivot[clases]
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=axes[idx], cbar_kws={"shrink": 0.8})
    axes[idx].set_title(f"{metrica} por Clase", fontsize=13, fontweight="bold")
    axes[idx].set_ylabel("" if idx > 0 else "Modelo")
    axes[idx].set_xlabel("Clase")
fig.suptitle("Heatmap de Metricas por Clase y Modelo", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, "20_heatmap_metricas_por_clase.png")

# [4/6] Barplot F1-Score por clase
print("[4/6] Barplot F1-Score por clase...")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(clases))
width = 0.25
for i, nombre in enumerate(nombres_modelo_exp):
    subset = df_por_clase[df_por_clase["Modelo"] == nombre]
    f1_vals = [subset[subset["Clase"] == c]["F1-Score"].values[0] for c in clases]
    bars = ax.bar(x + i * width, f1_vals, width, label=nombre,
                  color=colores_modelo_exp[nombre], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score por Clase y Modelo", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clases, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "21_barplot_f1_por_clase.png")

# [5/6] Radar chart comparativo
print("[5/6] Radar chart comparativo...")
metricas_radar = ["Accuracy", "Precision (weighted)", "Recall (weighted)",
                   "F1-Score (weighted)", "F1-Score (macro)"]
labels_radar = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
                "F1-Score\n(weighted)", "F1-Score\n(macro)"]
angles = np.linspace(0, 2 * np.pi, len(metricas_radar), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for nombre in nombres_modelo_exp:
    row = df_globales[df_globales["Modelo"] == nombre].iloc[0]
    valores = [row[m] for m in metricas_radar]
    valores += valores[:1]
    ax.plot(angles, valores, "o-", linewidth=2, markersize=6,
            label=nombre, color=colores_modelo_exp[nombre])
    ax.fill(angles, valores, alpha=0.1, color=colores_modelo_exp[nombre])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Perfil Comparativo de Modelos", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=10)
fig.tight_layout()
save_fig(fig, "22_radar_comparativo.png")

# [6/6] Tablas resumen visuales
print("[6/6] Tablas resumen visuales...")
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis("off")
header_tab = ["Modelo", "Accuracy", "Precision\n(weighted)", "Recall\n(weighted)",
              "F1-Score\n(weighted)", "Precision\n(macro)", "Recall\n(macro)",
              "F1-Score\n(macro)", "Tiempo (s)"]
cell_data_tab = []
for _, row in df_globales.iterrows():
    cell_data_tab.append([
        row["Modelo"], f"{row['Accuracy']:.4f}",
        f"{row['Precision (weighted)']:.4f}", f"{row['Recall (weighted)']:.4f}",
        f"{row['F1-Score (weighted)']:.4f}", f"{row['Precision (macro)']:.4f}",
        f"{row['Recall (macro)']:.4f}", f"{row['F1-Score (macro)']:.4f}",
        f"{row['Tiempo (s)']:.2f}",
    ])
table = ax.table(cellText=cell_data_tab, colLabels=header_tab, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
for j in range(len(header_tab)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(cell_data_tab) + 1):
    modelo_nombre = cell_data_tab[i - 1][0]
    if modelo_nombre == mejor_exp["Modelo"]:
        for j in range(len(header_tab)):
            table[i, j].set_facecolor("#d5f5e3")
            table[i, j].set_text_props(fontweight="bold")
    else:
        for j in range(len(header_tab)):
            table[i, j].set_facecolor("#eaf2f8" if i % 2 == 0 else "#ffffff")
ax.set_title(
    f"Tabla Resumen - Comparacion Experimental de Clasificadores\n"
    f"(Mejor modelo resaltado en verde: {mejor_exp['Modelo']})",
    fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "23_tabla_resumen_comparacion.png")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")
header_clase2 = ["Modelo", "Clase", "Precision", "Recall", "F1-Score", "Support"]
cell_clase2 = []
for _, row in df_por_clase.iterrows():
    cell_clase2.append([
        row["Modelo"], row["Clase"],
        f"{row['Precision']:.4f}", f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}", f"{int(row['Support']):,}"
    ])
table2 = ax.table(cellText=cell_clase2, colLabels=header_clase2, cellLoc="center", loc="center")
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2)
for j in range(len(header_clase2)):
    table2[0, j].set_facecolor("#2c3e50")
    table2[0, j].set_text_props(color="white", fontweight="bold")
color_filas_exp = {
    nombres_modelo_exp[0]: "#d6eaf8",
    nombres_modelo_exp[1]: "#fdebd0",
    nombres_modelo_exp[2]: "#d5f5e3",
}
for i in range(1, len(cell_clase2) + 1):
    bg = color_filas_exp.get(cell_clase2[i - 1][0], "#ffffff")
    for j in range(len(header_clase2)):
        table2[i, j].set_facecolor(bg)
ax.set_title("Metricas Detalladas por Clase y Modelo", fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "24_tabla_metricas_por_clase.png")

print(f"\n  MEJOR MODELO: {mejor_exp['Modelo']}")
print(f"  F1-Score (weighted): {mejor_exp['F1-Score (weighted)']:.4f}")
print("  Graficos generados: 7 (18 - 24)")


# =============================================================
# SECCION 5: PRUEBA DE MODELOS EN PRODUCCION
# =============================================================
# Equivalente a: scr/5_ProbarModelosProduccion.py
# =============================================================
print("\n" + "=" * 60)
print("  SECCION 5: PRUEBA DE MODELOS EN PRODUCCION")
print("=" * 60)

# --- 5.1 Cargar modelos y artefactos desde disco ---
print("\n  Cargando modelos exportados desde Models/...")
modelos_prod = {
    "Arbol de Decision": joblib.load(os.path.join(MODELS_DIR, "arbol_decision.pkl")),
    "SVM": joblib.load(os.path.join(MODELS_DIR, "svm.pkl")),
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
}
scaler_prod = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_target_prod = joblib.load(os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
le_sector_prod = joblib.load(os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
feature_cols_prod = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

print(f"  Modelos cargados: {list(modelos_prod.keys())}")
print(f"  Features esperadas: {feature_cols_prod}")
print(f"  Clases: {list(le_target_prod.classes_)}")

# --- 5.2 Cargar datos de prueba (JSON) ---
print("\n--- 5.2 CARGA DE DATOS DE PRUEBA ---")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data_json = json.load(f)
empresas = data_json["empresas"]
print(f"  Archivo: {JSON_PATH}")
print(f"  Empresas de prueba: {len(empresas)}")
print(f"\n  {'ID':<4} {'Empresa':<38} {'Sector':<20} {'Esperado':<10}")
print(f"  {'-' * 72}")
for emp in empresas:
    print(f"  {emp['id']:<4} {emp['nombre']:<38} {emp['Sector']:<20} {emp['desempeno_esperado']:<10}")

# --- 5.3 Preparar datos para inferencia ---
print("\n--- 5.3 PREPARACION DE DATOS ---")
df_test_prod = pd.DataFrame(empresas)
nombres_empresas = df_test_prod["nombre"].tolist()
ids_empresas = df_test_prod["id"].tolist()
esperados = df_test_prod["desempeno_esperado"].tolist()
df_test_prod["Sector"] = le_sector_prod.transform(df_test_prod["Sector"].astype(str))
X_prod = df_test_prod[feature_cols_prod].copy()
X_prod_scaled = pd.DataFrame(
    scaler_prod.transform(X_prod), columns=feature_cols_prod, index=X_prod.index
)
esperados_cod = le_target_prod.transform(esperados)
print(f"  Features preparadas: {X_prod_scaled.shape}")

# --- 5.4 Inferencia con cada modelo ---
print("\n--- 5.4 INFERENCIA - PREDICCIONES POR MODELO ---")
resultados_prod = {}
for nombre_modelo, modelo in modelos_prod.items():
    print(f"\n  --- {nombre_modelo} ---")
    y_pred_cod = modelo.predict(X_prod_scaled)
    y_pred_labels = le_target_prod.inverse_transform(y_pred_cod)
    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(X_prod_scaled)
        confianzas = np.max(probas, axis=1)
    else:
        confianzas = np.ones(len(y_pred_cod))
    aciertos = [pred == esp for pred, esp in zip(y_pred_labels, esperados)]
    resultados_prod[nombre_modelo] = {
        "predicciones": y_pred_labels,
        "confianzas": confianzas,
        "aciertos": aciertos,
    }
    print(f"  {'Empresa':<38} {'Esperado':<10} {'Prediccion':<12} {'Confianza':>10} {'Resultado':>10}")
    print(f"  {'-' * 80}")
    for i in range(len(empresas)):
        estado = "OK" if aciertos[i] else "FALLO"
        print(f"  {nombres_empresas[i]:<38} {esperados[i]:<10} {y_pred_labels[i]:<12} "
              f"{confianzas[i]:>9.1%} {estado:>10}")
    total_ok = sum(aciertos)
    print(f"\n  Aciertos: {total_ok}/{len(aciertos)} | Accuracy: {total_ok/len(aciertos):.1%} | "
          f"Confianza media: {np.mean(confianzas):.1%}")

# --- 5.5 Informe de eficiencia y confiabilidad ---
print("\n--- 5.5 INFORME DE EFICIENCIA Y CONFIABILIDAD ---")
informe_modelos = []
for nombre_modelo, res in resultados_prod.items():
    aciertos = res["aciertos"]
    confianzas = res["confianzas"]
    n = len(aciertos)
    n_ok = sum(aciertos)
    por_clase = {}
    for clase in le_target_prod.classes_:
        idx_clase = [i for i, e in enumerate(esperados) if e == clase]
        if idx_clase:
            por_clase[clase] = {
                "total": len(idx_clase),
                "aciertos": sum(res["aciertos"][i] for i in idx_clase),
                "accuracy": sum(res["aciertos"][i] for i in idx_clase) / len(idx_clase),
                "confianza_media": np.mean([res["confianzas"][i] for i in idx_clase]),
            }
    informe_modelos.append({
        "Modelo": nombre_modelo,
        "Empresas": n, "Aciertos": n_ok, "Fallos": n - n_ok,
        "Accuracy (%)": round(n_ok / n * 100, 2),
        "Confianza Media (%)": round(np.mean(confianzas) * 100, 2),
        "Confianza Min (%)": round(np.min(confianzas) * 100, 2),
        "Confianza Max (%)": round(np.max(confianzas) * 100, 2),
        "Detalle por clase": por_clase,
    })

df_informe = pd.DataFrame(informe_modelos)
print("\n  TABLA RESUMEN:")
print(df_informe[["Modelo", "Empresas", "Aciertos", "Fallos",
                   "Accuracy (%)", "Confianza Media (%)",
                   "Confianza Min (%)", "Confianza Max (%)"]].to_string(index=False))

# --- 5.6 Visualizaciones de produccion ---
print("\n--- 5.6 VISUALIZACIONES ---")
colores_prod = {
    "Arbol de Decision": "#3498db",
    "SVM": "#e67e22",
    "Random Forest": "#27ae60",
}

# [1/7] Accuracy por modelo
print("\n[1/7] Accuracy por modelo...")
fig, ax = plt.subplots(figsize=(10, 6))
nombres_p = list(resultados_prod.keys())
accuracies_p = [sum(r["aciertos"]) / len(r["aciertos"]) * 100 for r in resultados_prod.values()]
colores_p = [colores_prod[n] for n in nombres_p]
bars = ax.bar(nombres_p, accuracies_p, color=colores_p, edgecolor="black", linewidth=0.5, width=0.5)
for bar, val in zip(bars, accuracies_p):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy en Datos de Produccion (10 empresas de prueba)",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.4)
save_fig(fig, "25_accuracy_produccion.png")

# [2/7] Radar de confiabilidad por modelo
print("[2/7] Radar de confiabilidad por modelo...")
radar_metrics = {}
for nombre_modelo, res in resultados_prod.items():
    confs = res["confianzas"]
    aciertos_r = res["aciertos"]
    n = len(aciertos_r)
    acc_por_clase = {}
    for clase in le_target_prod.classes_:
        idx_c = [i for i, e in enumerate(esperados) if e == clase]
        if idx_c:
            acc_por_clase[clase] = sum(aciertos_r[i] for i in idx_c) / len(idx_c) * 100
    radar_metrics[nombre_modelo] = {
        "Accuracy\nGeneral": sum(aciertos_r) / n * 100,
        "Confianza\nMedia": np.mean(confs) * 100,
        "Confianza\nMinima": np.min(confs) * 100,
        "Acc. Clase\nAlto": acc_por_clase.get("Alto", 0),
        "Acc. Clase\nBajo": acc_por_clase.get("Bajo", 0),
        "Acc. Clase\nMedio": acc_por_clase.get("Medio", 0),
    }
metric_names_r = list(list(radar_metrics.values())[0].keys())
n_metrics_r = len(metric_names_r)
angles_r = np.linspace(0, 2 * np.pi, n_metrics_r, endpoint=False).tolist()
angles_r += angles_r[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for nombre_modelo in modelos_prod.keys():
    valores_r = [radar_metrics[nombre_modelo][m] for m in metric_names_r]
    valores_r += valores_r[:1]
    ax.plot(angles_r, valores_r, "o-", linewidth=2.5, markersize=8,
            label=nombre_modelo, color=colores_prod[nombre_modelo])
    ax.fill(angles_r, valores_r, alpha=0.12, color=colores_prod[nombre_modelo])
ax.set_xticks(angles_r[:-1])
ax.set_xticklabels(metric_names_r, fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="gray")
ax.set_title("Radar de Confiabilidad por Modelo\n(Accuracy, Confianza y Desempeno por Clase)",
             fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.02), fontsize=11,
          frameon=True, fancybox=True, shadow=True)
save_fig(fig, "26_radar_confiabilidad_modelos.png")

# [3/7] Radar de confianza por empresa
print("[3/7] Radar de confianza por empresa...")
fig, axes = plt.subplots(2, 5, figsize=(24, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()
empresa_angles = np.linspace(0, 2 * np.pi, len(modelos_prod), endpoint=False).tolist()
empresa_angles += empresa_angles[:1]
modelo_labels_r = [n.replace(" ", "\n") for n in modelos_prod.keys()]
for i, emp in enumerate(empresas):
    ax = axes[i]
    valores_emp = [resultados_prod[nm]["confianzas"][i] * 100 for nm in modelos_prod.keys()]
    valores_emp_plot = valores_emp + valores_emp[:1]
    ax.plot(empresa_angles, valores_emp_plot, "o-", linewidth=2, markersize=6, color="#2c3e50")
    ax.fill(empresa_angles, valores_emp_plot, alpha=0.2, color="#3498db")
    for j, nombre_modelo in enumerate(modelos_prod.keys()):
        ok = resultados_prod[nombre_modelo]["aciertos"][i]
        color_punto = "#27ae60" if ok else "#e74c3c"
        ax.plot(empresa_angles[j], valores_emp[j], "o", markersize=10, color=color_punto,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.set_xticks(empresa_angles[:-1])
    ax.set_xticklabels(modelo_labels_r, fontsize=7)
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")
    esperado_color = {"Alto": "#27ae60", "Medio": "#f39c12", "Bajo": "#e74c3c"}
    ax.set_title(f"{emp['id']}. {emp['nombre'][:20]}\nEsperado: {esperados[i]}",
                 fontsize=9, fontweight="bold", color=esperado_color[esperados[i]], pad=12)
fig.suptitle("Radar de Confianza por Empresa\n(Verde=Acierto, Rojo=Fallo en cada modelo)",
             fontsize=15, fontweight="bold", y=1.04)
fig.tight_layout()
save_fig(fig, "27_radar_confianza_por_empresa.png")

# [4/7] Heatmap de predicciones
print("[4/7] Heatmap de predicciones por empresa...")
fig, ax = plt.subplots(figsize=(14, 8))
heatmap_data = [[1 if resultados_prod[nm]["aciertos"][i] else 0
                 for nm in modelos_prod.keys()] for i in range(len(empresas))]
annot_data = [[f"{resultados_prod[nm]['predicciones'][i]}\n{resultados_prod[nm]['confianzas'][i]:.0%}"
               for nm in modelos_prod.keys()] for i in range(len(empresas))]
heatmap_df = pd.DataFrame(heatmap_data, columns=list(modelos_prod.keys()),
                           index=[f"{e['id']}. {e['nombre']}" for e in empresas])
annot_df = pd.DataFrame(annot_data, columns=list(modelos_prod.keys()), index=heatmap_df.index)
cmap = sns.color_palette(["#fadbd8", "#d5f5e3"], as_cmap=True)
sns.heatmap(heatmap_df, annot=annot_df, fmt="", cmap=cmap, linewidths=1,
            linecolor="white", ax=ax, cbar=False, annot_kws={"fontsize": 9})
for i, esp in enumerate(esperados):
    ax.text(len(modelos_prod) + 0.3, i + 0.5, esp, va="center", fontsize=10, fontweight="bold",
            color={"Alto": "#27ae60", "Medio": "#f39c12", "Bajo": "#e74c3c"}[esp])
ax.text(len(modelos_prod) + 0.3, -0.3, "Esperado", fontsize=10, fontweight="bold", color="black")
ax.set_title("Predicciones por Empresa y Modelo\n"
             "(Verde=Acierto, Rojo=Fallo | Anotacion: Prediccion + Confianza)",
             fontsize=13, fontweight="bold")
save_fig(fig, "28_heatmap_predicciones_empresa.png")

# [5/7] Accuracy por clase
print("[5/7] Accuracy por clase y modelo...")
fig, ax = plt.subplots(figsize=(12, 6))
clases_pres = sorted(set(esperados))
x = np.arange(len(clases_pres))
width = 0.25
for j, nombre_modelo in enumerate(modelos_prod.keys()):
    accs_clase = []
    for clase in clases_pres:
        idx = [i for i, e in enumerate(esperados) if e == clase]
        acc = sum(resultados_prod[nombre_modelo]["aciertos"][i] for i in idx) / len(idx) * 100 if idx else 0
        accs_clase.append(acc)
    bars = ax.bar(x + j * width, accs_clase, width, label=nombre_modelo,
                  color=colores_prod[nombre_modelo], edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, accs_clase):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy por Clase en Datos de Produccion", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clases_pres, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 120)
ax.grid(axis="y", linestyle="--", alpha=0.4)
save_fig(fig, "29_accuracy_por_clase_produccion.png")

# [6/7] Radar resumen ejecutivo
print("[6/7] Radar resumen ejecutivo...")
exec_metrics = ["Accuracy\nGeneral", "Confianza\nMedia", "Confianza\nMinima",
                "Consistencia\n(Aciertos/Total)", "Acc. Clase\nMayoritaria"]
exec_angles = np.linspace(0, 2 * np.pi, len(exec_metrics), endpoint=False).tolist()
exec_angles += exec_angles[:1]
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for nombre_modelo in modelos_prod.keys():
    info = [i for i in informe_modelos if i["Modelo"] == nombre_modelo][0]
    confs_exec = resultados_prod[nombre_modelo]["confianzas"]
    idx_bajo_exec = [i for i, e in enumerate(esperados) if e == "Bajo"]
    acc_bajo_exec = (sum(resultados_prod[nombre_modelo]["aciertos"][i] for i in idx_bajo_exec)
                     / len(idx_bajo_exec) * 100 if idx_bajo_exec else 0)
    vals_exec = [
        info["Accuracy (%)"], info["Confianza Media (%)"],
        info["Confianza Min (%)"], info["Aciertos"] / info["Empresas"] * 100, acc_bajo_exec
    ]
    vals_exec += vals_exec[:1]
    ax.plot(exec_angles, vals_exec, "o-", linewidth=2.5, markersize=8,
            label=nombre_modelo, color=colores_prod[nombre_modelo])
    ax.fill(exec_angles, vals_exec, alpha=0.1, color=colores_prod[nombre_modelo])
ax.set_xticks(exec_angles[:-1])
ax.set_xticklabels(exec_metrics, fontsize=10, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="gray")
ax.set_title("Resumen Ejecutivo: Eficiencia vs Confiabilidad",
             fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.02), fontsize=11,
          frameon=True, fancybox=True, shadow=True)
save_fig(fig, "30_radar_resumen_ejecutivo.png")

# [7/7] Tabla visual del informe completo
print("[7/7] Tabla visual del informe...")
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis("off")
header_prod = ["ID", "Empresa", "Esperado",
               "Arbol\nPrediccion", "Arbol\nConfianza",
               "SVM\nPrediccion", "SVM\nConfianza",
               "RF\nPrediccion", "RF\nConfianza", "Consenso"]
cell_data_prod = []
for i in range(len(empresas)):
    row = [str(ids_empresas[i]), nombres_empresas[i][:25], esperados[i]]
    coincidencias_prod = 0
    for nombre_modelo in modelos_prod.keys():
        pred = resultados_prod[nombre_modelo]["predicciones"][i]
        conf = resultados_prod[nombre_modelo]["confianzas"][i]
        row.append(pred)
        row.append(f"{conf:.1%}")
        if resultados_prod[nombre_modelo]["aciertos"][i]:
            coincidencias_prod += 1
    row.append(f"{coincidencias_prod}/{len(modelos_prod)}")
    cell_data_prod.append(row)
table_prod = ax.table(cellText=cell_data_prod, colLabels=header_prod, cellLoc="center", loc="center")
table_prod.auto_set_font_size(False)
table_prod.set_fontsize(8)
table_prod.scale(1, 2)
for j in range(len(header_prod)):
    table_prod[0, j].set_facecolor("#2c3e50")
    table_prod[0, j].set_text_props(color="white", fontweight="bold")
pred_col_indices_prod = [3, 5, 7]
modelo_names_prod = list(modelos_prod.keys())
for i in range(len(cell_data_prod)):
    for k, col_idx in enumerate(pred_col_indices_prod):
        nombre_m = modelo_names_prod[k]
        if resultados_prod[nombre_m]["aciertos"][i]:
            table_prod[i + 1, col_idx].set_facecolor("#d5f5e3")
        else:
            table_prod[i + 1, col_idx].set_facecolor("#fadbd8")
    consenso_str = cell_data_prod[i][-1]
    n_ok_c = int(consenso_str.split("/")[0])
    if n_ok_c == 3:
        table_prod[i + 1, len(header_prod) - 1].set_facecolor("#27ae60")
        table_prod[i + 1, len(header_prod) - 1].set_text_props(color="white", fontweight="bold")
    elif n_ok_c >= 2:
        table_prod[i + 1, len(header_prod) - 1].set_facecolor("#f1c40f")
    else:
        table_prod[i + 1, len(header_prod) - 1].set_facecolor("#e74c3c")
        table_prod[i + 1, len(header_prod) - 1].set_text_props(color="white", fontweight="bold")
ax.set_title("Informe de Produccion: Predicciones, Confianza y Consenso por Empresa",
             fontsize=14, fontweight="bold", pad=25)
save_fig(fig, "31_informe_produccion_completo.png")

# Resumen final de produccion
print("\n" + "=" * 60)
print("  RESUMEN FINAL - PRUEBA EN PRODUCCION")
print("=" * 60)
print(f"\n  {'Modelo':<22} {'Accuracy':>10} {'Confianza':>12} {'Aciertos':>10}")
print(f"  {'-' * 54}")
mejor_acc_prod = max(i["Accuracy (%)"] for i in informe_modelos)
for info in informe_modelos:
    marca = " <--" if info["Accuracy (%)"] == mejor_acc_prod else ""
    print(f"  {info['Modelo']:<22} {info['Accuracy (%)']:>9.1f}% "
          f"{info['Confianza Media (%)']:>10.1f}%  "
          f"{info['Aciertos']}/{info['Empresas']}{marca}")
consenso_total = sum(
    1 for i in range(len(empresas))
    if all(resultados_prod[m]["aciertos"][i] for m in modelos_prod)
)
print(f"\n  Empresas con consenso unanime (3/3): {consenso_total}/{len(empresas)}")
print("  Graficos generados: 7 (25 - 31)")


# =============================================================
# RESUMEN GLOBAL DEL PROYECTO
# =============================================================
print("\n" + "=" * 60)
print("  RESUMEN GLOBAL DEL PROYECTO")
print("  Pipeline completo ejecutado exitosamente")
print("=" * 60)
print(f"\n  Dataset:      {DATA_PATH}")
print(f"  Registros:    {X_scaled.shape[0] + X_train.shape[0] + X_test.shape[0]:,} totales")
print(f"  Features:     {len(feature_cols)}")
print(f"  Clases:       {clases}")
print(f"\n  ETAPAS COMPLETADAS:")
print(f"    [1] EDA:                    8 graficos  (01 - 08)")
print(f"    [2] Preprocesamiento:       4 graficos  (09 - 12)")
print(f"    [3] Entrenamiento:          5 graficos  (13 - 17)")
print(f"    [4] Comparacion:            7 graficos  (18 - 24)")
print(f"    [5] Produccion:             7 graficos  (25 - 31)")
print(f"\n  Total de graficos generados: 31")
print(f"  Resultados en: {RESULTS_DIR}")
print(f"  Modelos en:    {MODELS_DIR}")
print(f"\n  MEJOR MODELO (prueba): {mejor_exp['Modelo']}")
print(f"    F1-Score (weighted): {mejor_exp['F1-Score (weighted)']:.4f}")
print("=" * 60)
print("\nPipeline completo ejecutado exitosamente.")
