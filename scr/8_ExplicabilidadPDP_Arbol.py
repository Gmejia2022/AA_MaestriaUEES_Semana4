# -*- coding: utf-8 -*-
"""
8_ExplicabilidadPDP_Arbol.py - Explicabilidad con PDP y Visualizacion de Arbol

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan
  - Ingeniero David Perugachi Rojas

Objetivo:
  Aplicar tecnicas de explicabilidad (XAI) complementarias:

  Tecnica 3 - Partial Dependence Plots (PDP):
    - PDP de las 4 features mas importantes sobre Random Forest
    - PDP bidimensional (interaccion entre 2 features clave)
    - Individual Conditional Expectation (ICE) sobre una feature clave
    - Comparativa PDP entre los 3 modelos para la feature mas importante

  Tecnica 4 - Visualizacion detallada del Arbol de Decision:
    - Arbol completo (profundidad 3) con estadisticas por nodo
    - Analisis de reglas de decision extraidas del arbol
    - Tabla de reglas: feature, umbral, muestras, impureza
    - Profundidad vs accuracy: efecto de podar el arbol

  Prerequisito: scikit-learn >= 1.3.0 (incluye PartialDependenceDisplay)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla (necesario en entornos sin display)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.max_open_warning": 0})

# =========================================================
# 1. CONFIGURACION DE RUTAS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, name):
    """Guarda una figura en la carpeta results."""
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA DE MODELOS Y DATOS
# =========================================================
print("=" * 60)
print("  EXPLICABILIDAD XAI - PDP Y VISUALIZACION DE ARBOL")
print("  Dataset: Empresas del Ecuador - 2024")
print("=" * 60)

print("\n  Cargando modelos exportados...")
dt_model = joblib.load(os.path.join(MODELS_DIR, "arbol_decision.pkl"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_target = joblib.load(os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
le_sector = joblib.load(os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

clases = list(le_target.classes_)
print(f"  Modelos cargados: Arbol de Decision, SVM, Random Forest")
print(f"  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Clases: {clases}")

# =========================================================
# 3. RECONSTRUCCION DEL CONJUNTO DE DATOS
# =========================================================
print("\n  Reconstruyendo conjuntos de entrenamiento y prueba...")

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
df.columns = (
    df.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)

col_ano = [c for c in df.columns if df[c].nunique() == 1 and df[c].dtype in ["int64", "float64"]]
if col_ano:
    df = df.drop(columns=col_ano)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Margen_Neto"])
df["Desempeno"] = pd.qcut(df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")

df["Sector"] = le_sector.transform(df["Sector"].astype(str))
df["Desempeno_cod"] = le_target.fit_transform(df["Desempeno"])

X = df[feature_cols]
y = df["Desempeno_cod"]

X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Muestra reducida para PDP (costoso con datasets grandes)
PDP_SAMPLE = min(3000, len(X_test))
X_pdp = X_test.sample(n=PDP_SAMPLE, random_state=42)
y_pdp = y_test.loc[X_pdp.index]

print(f"  Entrenamiento: {X_train.shape[0]:,} | Prueba: {X_test.shape[0]:,}")
print(f"  Muestra para PDP: {PDP_SAMPLE:,} registros")

# Features mas importantes segun Random Forest (Gini)
importancias_rf = rf_model.feature_importances_
feat_imp_order = np.argsort(importancias_rf)[::-1]
top4_features = [feature_cols[i] for i in feat_imp_order[:4]]
top4_idx = feat_imp_order[:4].tolist()
top1_feature = top4_features[0]
top1_idx = top4_idx[0]
top2_feature = top4_features[1]
top2_idx = top4_idx[1]

print(f"\n  Top 4 features (RF Gini): {top4_features}")

palette_clase = {"Alto": "#27ae60", "Bajo": "#e74c3c", "Medio": "#f39c12"}
colores_modelo = {
    "Arbol de Decision": "#3498db",
    "SVM": "#e67e22",
    "Random Forest": "#27ae60",
}
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)

# =========================================================
# 4. TECNICA 3: PARTIAL DEPENDENCE PLOTS (PDP)
# =========================================================
print("\n" + "=" * 60)
print("  TECNICA 3: PARTIAL DEPENDENCE PLOTS (PDP)")
print("  Modelo base: Random Forest")
print("=" * 60)

# --- 4.1 PDP de las 4 features mas importantes (una clase por vez) ---
print("\n[1/4] PDP de las 4 features mas importantes (clase 'Alto')...")

# Clase "Alto" = indice en le_target.classes_
idx_alto = clases.index("Alto")
idx_bajo = clases.index("Bajo")
idx_medio = clases.index("Medio")

fig, axes = plt.subplots(3, 4, figsize=(22, 16))

for row, (clase_nombre, clase_idx) in enumerate(zip(clases, [idx_alto, idx_bajo, idx_medio])):
    for col, (feat_nombre, feat_idx) in enumerate(zip(top4_features, top4_idx)):
        ax = axes[row, col]
        pd_result = partial_dependence(
            rf_model, X_pdp, features=[feat_idx],
            kind="average", grid_resolution=50
        )
        grid_vals = pd_result["grid_values"][0]
        # sklearn >= 1.4 multiclase: average shape (n_clases, n_grid)
        avg_raw = pd_result["average"]
        avg_pred = avg_raw[clase_idx] if avg_raw.ndim == 2 else avg_raw[0]

        ax.plot(grid_vals, avg_pred, color=palette_clase[clase_nombre],
                linewidth=2.5, label=f"PDP - {clase_nombre}")
        ax.fill_between(grid_vals, avg_pred, alpha=0.15, color=palette_clase[clase_nombre])
        ax.axhline(y=avg_pred.mean(), color="gray", linewidth=1,
                   linestyle="--", alpha=0.7, label=f"Media: {avg_pred.mean():.3f}")

        ax.set_xlabel(f"{feat_nombre} (escalado)", fontsize=9)
        ax.set_ylabel("Probabilidad predicha" if col == 0 else "", fontsize=9)
        ax.set_title(f"{feat_nombre}\nClase: {clase_nombre}",
                     fontsize=10, fontweight="bold", color=palette_clase[clase_nombre])
        ax.legend(fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)

fig.suptitle("Partial Dependence Plots (PDP) - Top 4 Features x 3 Clases\n"
             "(Random Forest | Muestra de prueba)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, "40_pdp_top4_features_por_clase.png")

# --- 4.2 PDP bidimensional: interaccion entre las 2 features mas importantes ---
print("[2/4] PDP bidimensional - interaccion entre top 2 features...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (clase_nombre, clase_idx) in zip(axes, zip(clases, [idx_alto, idx_bajo, idx_medio])):
    pd_2d = partial_dependence(
        rf_model, X_pdp,
        features=[(top1_idx, top2_idx)],
        kind="average",
        grid_resolution=20
    )
    # sklearn >= 1.4 multiclase: average shape (n_clases, n_grid1, n_grid2)
    avg_2d = pd_2d["average"]
    Z = avg_2d[clase_idx] if avg_2d.ndim == 3 else avg_2d[0]
    xx = pd_2d["grid_values"][0]
    yy = pd_2d["grid_values"][1]

    XX, YY = np.meshgrid(xx, yy)
    im = ax.contourf(XX, YY, Z.T, levels=20, cmap="RdYlGn")
    ax.contour(XX, YY, Z.T, levels=10, colors="black", linewidths=0.3, alpha=0.4)
    plt.colorbar(im, ax=ax, label="Probabilidad predicha")

    ax.set_xlabel(f"{top1_feature} (escalado)", fontsize=10)
    ax.set_ylabel(f"{top2_feature} (escalado)", fontsize=10)
    ax.set_title(f"Clase: {clase_nombre}",
                 fontsize=12, fontweight="bold", color=palette_clase[clase_nombre])

fig.suptitle(f"PDP Bidimensional: {top1_feature} x {top2_feature}\n"
             "(Interaccion entre las 2 features mas importantes | Random Forest)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "41_pdp_bidimensional_interaccion.png")

# --- 4.3 ICE (Individual Conditional Expectation) para la feature top1 ---
print("[3/4] ICE Plot - Individual Conditional Expectation...")

ICE_SAMPLE = min(300, len(X_pdp))
X_ice = X_pdp.sample(n=ICE_SAMPLE, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, (clase_nombre, clase_idx) in zip(axes, zip(clases, [idx_alto, idx_bajo, idx_medio])):
    pd_ice = partial_dependence(
        rf_model, X_ice,
        features=[top1_idx],
        kind="both",
        grid_resolution=40
    )
    grid_vals = pd_ice["grid_values"][0]
    # sklearn >= 1.4: individual shape (n_clases, n_muestras, n_grid), average (n_clases, n_grid)
    ind_raw = pd_ice["individual"]
    avg_raw = pd_ice["average"]
    ice_lines = ind_raw[clase_idx] if ind_raw.ndim == 3 else ind_raw[0]
    avg_line = avg_raw[clase_idx] if avg_raw.ndim == 2 else avg_raw[0]

    # Trazar lineas ICE individuales (transparentes)
    for line in ice_lines:
        ax.plot(grid_vals, line, color=palette_clase[clase_nombre],
                linewidth=0.4, alpha=0.15)

    # Trazar PDP promedio encima
    ax.plot(grid_vals, avg_line, color="black",
            linewidth=2.5, label="PDP (promedio)", zorder=5)

    ax.set_xlabel(f"{top1_feature} (escalado)", fontsize=10)
    ax.set_ylabel("Probabilidad predicha" if ax == axes[0] else "", fontsize=10)
    ax.set_title(f"Clase: {clase_nombre}\n({ICE_SAMPLE} curvas ICE individuales)",
                 fontsize=11, fontweight="bold", color=palette_clase[clase_nombre])
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

fig.suptitle(f"ICE Plot: {top1_feature}\n"
             "(Lineas de color = curvas individuales | Linea negra = PDP promedio | Random Forest)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "42_ice_plot_top_feature.png")

# --- 4.4 PDP comparativo entre los 3 modelos para top1_feature, clase "Alto" ---
print("[4/4] PDP comparativo entre los 3 modelos (feature mas importante)...")

modelos_pdp = {
    "Arbol de Decision": dt_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, clase_nombre in zip(axes, clases):
    clase_idx = clases.index(clase_nombre)
    for nombre_mod, modelo in modelos_pdp.items():
        try:
            pd_res = partial_dependence(
                modelo, X_pdp,
                features=[top1_idx],
                kind="average",
                grid_resolution=40
            )
            grid_vals = pd_res["grid_values"][0]
            avg_raw = pd_res["average"]
            avg_pred = avg_raw[clase_idx] if avg_raw.ndim == 2 else avg_raw[0]
            ax.plot(grid_vals, avg_pred, color=colores_modelo[nombre_mod],
                    linewidth=2.5, label=nombre_mod)
            ax.fill_between(grid_vals, avg_pred, alpha=0.08,
                            color=colores_modelo[nombre_mod])
        except Exception:
            # SVM puede no soportar predict_proba con todos los kernels
            pass

    ax.set_xlabel(f"{top1_feature} (escalado)", fontsize=10)
    ax.set_ylabel("Probabilidad predicha" if ax == axes[0] else "", fontsize=10)
    ax.set_title(f"Clase: {clase_nombre}",
                 fontsize=12, fontweight="bold", color=palette_clase[clase_nombre])
    ax.legend(fontsize=9, loc="best")
    ax.grid(linestyle="--", alpha=0.4)

fig.suptitle(f"PDP Comparativo - {top1_feature} (3 modelos)\n"
             "(Como cambia la probabilidad de cada clase al variar la feature mas importante)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "43_pdp_comparativo_modelos.png")

# =========================================================
# 5. TECNICA 4: VISUALIZACION DETALLADA DEL ARBOL DE DECISION
# =========================================================
print("\n" + "=" * 60)
print("  TECNICA 4: VISUALIZACION DETALLADA DEL ARBOL DE DECISION")
print("=" * 60)

# --- 5.1 Arbol completo con estadisticas de nodos (profundidad 4) ---
print("\n[1/4] Arbol de Decision detallado (profundidad 4)...")

fig, ax = plt.subplots(figsize=(28, 14))
plot_tree(
    dt_model,
    max_depth=4,
    feature_names=feature_cols,
    class_names=clases,
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax,
    proportion=False,
    impurity=True,
    precision=3
)
ax.set_title("Arbol de Decision - Estructura Detallada (primeros 4 niveles)\n"
             "Cada nodo muestra: feature | umbral | impureza Gini | muestras | distribucion de clases",
             fontsize=14, fontweight="bold")
save_fig(fig, "44_arbol_decision_detallado.png")

# --- 5.2 Reglas de decision extraidas del arbol ---
print("[2/4] Extraccion y visualizacion de reglas de decision...")

reglas_texto = export_text(dt_model, feature_names=feature_cols, max_depth=4)

# Analizar la estructura del arbol para extraer reglas estadisticas
tree_ = dt_model.tree_
feature_names_arr = np.array(feature_cols)

# Recopilar informacion de todos los nodos internos
nodos_internos = []
for node_id in range(tree_.node_count):
    if tree_.children_left[node_id] != tree_.children_left[0] or node_id == 0:
        if tree_.feature[node_id] >= 0:  # nodo interno (no hoja)
            nodos_internos.append({
                "Nodo": node_id,
                "Feature": feature_names_arr[tree_.feature[node_id]],
                "Umbral": round(tree_.threshold[node_id], 4),
                "Muestras": tree_.n_node_samples[node_id],
                "Gini": round(tree_.impurity[node_id], 4),
                "Profundidad": int(np.floor(np.log2(node_id + 1))) if node_id > 0 else 0,
            })

df_nodos = pd.DataFrame(nodos_internos).head(20)  # primeros 20 nodos

# Visualizacion: tabla de reglas + distribucion de features usadas
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Tabla de nodos
ax_tabla = fig.add_subplot(gs[0, :])
ax_tabla.axis("off")

tabla = ax_tabla.table(
    cellText=df_nodos.values,
    colLabels=df_nodos.columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.08, 0.25, 0.12, 0.12, 0.10, 0.12]
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1, 1.6)

for j in range(len(df_nodos.columns)):
    tabla[0, j].set_facecolor("#2c3e50")
    tabla[0, j].set_text_props(color="white", fontweight="bold")

colores_fila = ["#eaf2f8" if i % 2 == 0 else "#ffffff" for i in range(len(df_nodos))]
for i, color in enumerate(colores_fila):
    for j in range(len(df_nodos.columns)):
        tabla[i + 1, j].set_facecolor(color)

ax_tabla.set_title("Primeros 20 Nodos Internos del Arbol de Decision\n"
                   "(Feature | Umbral | Muestras | Impureza Gini | Profundidad)",
                   fontsize=12, fontweight="bold", pad=20)

# Frecuencia de uso de cada feature en el arbol
ax_freq = fig.add_subplot(gs[1, 0])
feature_freq = pd.Series(feature_names_arr[tree_.feature[tree_.feature >= 0]]).value_counts()
colors_freq = plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_freq)))
bars = ax_freq.barh(feature_freq.index, feature_freq.values,
                    color=colors_freq, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, feature_freq.values):
    ax_freq.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9, fontweight="bold")
ax_freq.set_xlabel("Numero de nodos que usan la feature", fontsize=10)
ax_freq.set_title("Frecuencia de Uso de Features en el Arbol", fontsize=11, fontweight="bold")
ax_freq.grid(axis="x", linestyle="--", alpha=0.4)

# Gini promedio por profundidad
ax_gini = fig.add_subplot(gs[1, 1])
gini_por_prof = df_nodos.groupby("Profundidad")["Gini"].mean()
ax_gini.plot(gini_por_prof.index, gini_por_prof.values, "o-",
             color="#e74c3c", linewidth=2.5, markersize=8)
ax_gini.fill_between(gini_por_prof.index, gini_por_prof.values,
                     alpha=0.2, color="#e74c3c")
ax_gini.set_xlabel("Profundidad del nodo", fontsize=10)
ax_gini.set_ylabel("Impureza Gini promedio", fontsize=10)
ax_gini.set_title("Reduccion de Impureza Gini por Profundidad\n"
                  "(A mayor profundidad, menor impureza = mas puro)",
                  fontsize=11, fontweight="bold")
ax_gini.grid(linestyle="--", alpha=0.4)

fig.suptitle("Analisis Estructural del Arbol de Decision",
             fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, "45_arbol_analisis_nodos.png")

# --- 5.3 Efecto de la profundidad maxima en la precision ---
print("[3/4] Analisis de profundidad vs accuracy (poda del arbol)...")

profundidades = list(range(1, 21))
train_accs, test_accs, train_f1s, test_f1s = [], [], [], []

for prof in profundidades:
    dt_temp = DecisionTreeClassifier(
        max_depth=prof,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )
    dt_temp.fit(X_train, y_train)

    y_train_pred = dt_temp.predict(X_train)
    y_test_pred = dt_temp.predict(X_test)

    train_accs.append(accuracy_score(y_train, y_train_pred))
    test_accs.append(accuracy_score(y_test, y_test_pred))
    train_f1s.append(f1_score(y_train, y_train_pred, average="weighted"))
    test_f1s.append(f1_score(y_test, y_test_pred, average="weighted"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
axes[0].plot(profundidades, train_accs, "o-", color="#3498db",
             linewidth=2, markersize=6, label="Entrenamiento")
axes[0].plot(profundidades, test_accs, "s-", color="#e74c3c",
             linewidth=2, markersize=6, label="Prueba")
axes[0].axvline(x=dt_model.get_depth(), color="green", linewidth=1.5,
                linestyle="--", label=f"Profundidad actual ({dt_model.get_depth()})")
axes[0].set_xlabel("Profundidad maxima del arbol", fontsize=11)
axes[0].set_ylabel("Accuracy", fontsize=11)
axes[0].set_title("Accuracy vs Profundidad del Arbol", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].set_ylim(0.5, 1.02)
axes[0].grid(linestyle="--", alpha=0.4)

# F1-Score
axes[1].plot(profundidades, train_f1s, "o-", color="#3498db",
             linewidth=2, markersize=6, label="Entrenamiento")
axes[1].plot(profundidades, test_f1s, "s-", color="#e74c3c",
             linewidth=2, markersize=6, label="Prueba")
axes[1].axvline(x=dt_model.get_depth(), color="green", linewidth=1.5,
                linestyle="--", label=f"Profundidad actual ({dt_model.get_depth()})")

# Marcar la profundidad optima (max F1 en prueba)
opt_prof = profundidades[np.argmax(test_f1s)]
axes[1].axvline(x=opt_prof, color="orange", linewidth=1.5,
                linestyle=":", label=f"Optima prueba (prof={opt_prof})")
axes[1].set_xlabel("Profundidad maxima del arbol", fontsize=11)
axes[1].set_ylabel("F1-Score (weighted)", fontsize=11)
axes[1].set_title("F1-Score vs Profundidad del Arbol", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].set_ylim(0.5, 1.02)
axes[1].grid(linestyle="--", alpha=0.4)

fig.suptitle("Efecto de la Profundidad del Arbol de Decision en el Rendimiento\n"
             "(Analisis de sobreajuste - underfitting vs overfitting)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "46_arbol_profundidad_vs_accuracy.png")

print(f"\n  Profundidad actual del modelo: {dt_model.get_depth()}")
print(f"  Profundidad optima en prueba:  {opt_prof} (F1 = {max(test_f1s):.4f})")

# --- 5.4 Comparativa visual: importancia Gini (arbol) vs RF vs PDP ---
print("[4/4] Resumen visual comparativo de las 4 tecnicas XAI...")

# Recopilar importancias de las 4 tecnicas
imp_gini_dt = pd.Series(dt_model.feature_importances_, index=feature_cols)
imp_gini_rf = pd.Series(rf_model.feature_importances_, index=feature_cols)

# PDP: aproximar importancia como rango de prediccion (max - min del PDP)
imp_pdp = {}
for feat_idx, feat_nombre in zip(top4_idx, top4_features):
    try:
        pd_res = partial_dependence(
            rf_model, X_pdp, features=[feat_idx],
            kind="average", grid_resolution=30
        )
        avg_r = pd_res["average"]
        avg_alto = avg_r[idx_alto] if avg_r.ndim == 2 else avg_r[0]
        rango = avg_alto.max() - avg_alto.min()
        imp_pdp[feat_nombre] = rango
    except Exception:
        imp_pdp[feat_nombre] = 0.0

# Rellenar features que no estan en top4 con 0
for f in feature_cols:
    if f not in imp_pdp:
        imp_pdp[f] = 0.0
imp_pdp_series = pd.Series(imp_pdp)

# Normalizar todo a [0,1]
def normalizar(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

comp_4tec = pd.DataFrame({
    "Gini (Arbol)": normalizar(imp_gini_dt),
    "Gini (RF)": normalizar(imp_gini_rf),
    "PDP Rango": normalizar(imp_pdp_series),
}).sort_values("Gini (RF)", ascending=False)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(comp_4tec))
width = 0.28
colores_tec = ["#3498db", "#27ae60", "#9b59b6"]
labels_tec = ["Gini Impurity\n(Arbol de Decision)", "Gini Impurity\n(Random Forest)", "PDP Rango\n(Random Forest)"]

for i, (col, lbl) in enumerate(zip(comp_4tec.columns, labels_tec)):
    bars = ax.bar(x + i * width, comp_4tec[col], width,
                  label=lbl, color=colores_tec[i], edgecolor="black",
                  linewidth=0.5, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(comp_4tec.index, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Importancia normalizada [0, 1]", fontsize=11)
ax.set_title("Comparativa de Importancia de Features segun Distintas Tecnicas XAI\n"
             "(Todas las metricas normalizadas a [0,1] para comparacion directa)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "47_comparativa_tecnicas_xai.png")

# =========================================================
# 6. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DE EXPLICABILIDAD - PDP Y ARBOL DE DECISION")
print("=" * 60)

print("\n  TECNICA 3 - Partial Dependence Plots (PDP):")
print(f"    Feature principal analizada: {top1_feature}")
print(f"    Feature secundaria (PDP 2D): {top2_feature}")
print(f"    Muestras utilizadas: {PDP_SAMPLE:,}")
print(f"    Curvas ICE individuales: {ICE_SAMPLE}")

print("\n  TECNICA 4 - Arbol de Decision:")
print(f"    Profundidad actual del modelo: {dt_model.get_depth()}")
print(f"    Numero de hojas:              {dt_model.get_n_leaves()}")
print(f"    Nodos internos analizados:    {len(nodos_internos)}")
print(f"    Profundidad optima (F1 max):  {opt_prof} (F1 = {max(test_f1s):.4f})")
print(f"    Feature mas usada en nodos:  {feature_freq.index[0]} ({feature_freq.iloc[0]} veces)")

print("\n  Graficos generados:")
graficos = [
    "40_pdp_top4_features_por_clase.png",
    "41_pdp_bidimensional_interaccion.png",
    "42_ice_plot_top_feature.png",
    "43_pdp_comparativo_modelos.png",
    "44_arbol_decision_detallado.png",
    "45_arbol_analisis_nodos.png",
    "46_arbol_profundidad_vs_accuracy.png",
    "47_comparativa_tecnicas_xai.png",
]
for g in graficos:
    print(f"    - {g}")

print(f"\n  Imagenes en: {RESULTS_DIR}")
print("=" * 60)
print("\nExplicabilidad PDP y Arbol de Decision completada.")
