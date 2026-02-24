# -*- coding: utf-8 -*-
"""
7_ExplicabilidadSHAP.py - Explicabilidad con SHAP y Permutation Feature Importance

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan
  - Ingeniero David Perugachi Rojas

Objetivo:
  Aplicar tecnicas de explicabilidad (XAI) a los modelos entrenados:

  Tecnica 1 - SHAP (SHapley Additive exPlanations):
    - SHAP values para Random Forest (mejor modelo global)
    - Beeswarm plot: impacto de cada feature en todas las predicciones
    - Bar plot: importancia media absoluta de features (SHAP)
    - Waterfall plot: explicacion de una prediccion individual
    - Dependence plot: relacion entre una feature y su impacto SHAP

  Tecnica 2 - Permutation Feature Importance:
    - Calculo de importancia por permutacion para los 3 modelos
    - Comparativa de importancias entre modelos
    - Analisis de robustez (media +/- std de cada feature)

  Prerequisito: pip install shap
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

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
print("  EXPLICABILIDAD XAI - SHAP Y PERMUTATION IMPORTANCE")
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
print(f"  Features: {feature_cols}")
print(f"  Clases: {clases}")

# =========================================================
# 3. RECONSTRUCCION DEL CONJUNTO DE PRUEBA
# =========================================================
print("\n  Reconstruyendo conjunto de prueba...")

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig", engine="python", on_bad_lines="skip")
df.columns = (
    df.columns.str.strip()
    .str.replace("\n", "_", regex=False)
    .str.replace(" ", "_", regex=False)
    .str.replace(".", "", regex=False)
)

# Eliminar columna Ano
col_ano = [c for c in df.columns if df[c].nunique() == 1 and df[c].dtype in ["int64", "float64"]]
if col_ano:
    df = df.drop(columns=col_ano)

# Convertir columnas numericas
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in df.columns:
    if col not in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Variable objetivo
epsilon = 1e-7
df["Margen_Neto"] = df["UtilidadNeta"] / (df["IngresosTotales"] + epsilon)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Margen_Neto"])
df["Desempeno"] = pd.qcut(df["Margen_Neto"], q=3, labels=["Bajo", "Medio", "Alto"], duplicates="drop")

# Codificar
df["Sector"] = le_sector.transform(df["Sector"].astype(str))
df["Desempeno_cod"] = le_target.fit_transform(df["Desempeno"])

X = df[feature_cols]
y = df["Desempeno_cod"]

X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

_, X_test, _, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Muestra para SHAP (costoso computacionalmente con datasets grandes)
SHAP_SAMPLE = min(2000, len(X_test))
X_shap = X_test.sample(n=SHAP_SAMPLE, random_state=42)

print(f"  Conjunto de prueba: {X_test.shape[0]:,} registros")
print(f"  Muestra para SHAP:  {SHAP_SAMPLE:,} registros")

# =========================================================
# 4. TECNICA 1: SHAP (SHapley Additive exPlanations)
# =========================================================
print("\n" + "=" * 60)
print("  TECNICA 1: SHAP - SHapley Additive exPlanations")
print("  Modelo: Random Forest (mejor modelo global)")
print("=" * 60)

print("\n  Calculando SHAP values para Random Forest...")
print("  (Usando TreeExplainer - optimizado para arboles de decision)")

# TreeExplainer es el mas eficiente para modelos basados en arboles
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_raw = explainer_rf.shap_values(X_shap)

# Normalizar estructura: SHAP moderno puede retornar array 3D (n_muestras, n_features, n_clases)
# o lista de arrays [(n_muestras, n_features)] x n_clases
# Convertimos siempre a lista de arrays: shap_values_rf[i] = (n_muestras, n_features)
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    # shape: (n_muestras, n_features, n_clases) -> lista de (n_muestras, n_features)
    n_clases_shap = shap_values_raw.shape[2]
    shap_values_rf = [shap_values_raw[:, :, i] for i in range(n_clases_shap)]
elif isinstance(shap_values_raw, list):
    shap_values_rf = shap_values_raw
else:
    # array 2D (clasificacion binaria o multioutput plano): envolver en lista
    shap_values_rf = [shap_values_raw]

print(f"  SHAP values calculados.")
print(f"  Shape: {len(shap_values_rf)} clases x {shap_values_rf[0].shape[0]} muestras x {shap_values_rf[0].shape[1]} features")

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.05)
palette_clase = {"Alto": "#27ae60", "Bajo": "#e74c3c", "Medio": "#f39c12"}

# --- 4.1 SHAP Bar Plot: importancia media absoluta global (todas las clases) ---
print("\n[1/5] SHAP Bar Plot - Importancia media absoluta global...")

# Media absoluta de SHAP values promediando todas las clases
mean_abs_shap = np.mean([np.abs(shap_values_rf[i]).mean(axis=0) for i in range(len(clases))], axis=0)
shap_importance = pd.DataFrame({
    "Feature": feature_cols,
    "SHAP_mean_abs": mean_abs_shap
}).sort_values("SHAP_mean_abs", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = plt.cm.RdYlGn(shap_importance["SHAP_mean_abs"] / shap_importance["SHAP_mean_abs"].max())
bars = ax.barh(shap_importance["Feature"], shap_importance["SHAP_mean_abs"],
               color=colors_bar, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, shap_importance["SHAP_mean_abs"]):
    ax.text(val + shap_importance["SHAP_mean_abs"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Media del valor SHAP absoluto (impacto en la prediccion)", fontsize=11)
ax.set_title("SHAP - Importancia Global de Features\n(Random Forest | Promedio todas las clases)",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)
fig.tight_layout()
save_fig(fig, "32_shap_barplot_global.png")

# --- 4.2 SHAP Beeswarm Plot por clase ---
print("[2/5] SHAP Beeswarm Plot por clase...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for i, (clase, ax) in enumerate(zip(clases, axes)):
    # Ordenar features por importancia SHAP para esta clase
    mean_abs_clase = np.abs(shap_values_rf[i]).mean(axis=0)
    order_idx = np.argsort(mean_abs_clase)[::-1]
    top_n = min(8, len(feature_cols))
    top_idx = order_idx[:top_n]

    shap_vals_clase = shap_values_rf[i][:, top_idx]
    feat_vals_clase = X_shap.iloc[:, top_idx]
    feat_names_top = [feature_cols[j] for j in top_idx]

    # Grafico de puntos coloreados por valor de feature
    for k, feat_name in enumerate(feat_names_top):
        y_pos = top_n - 1 - k
        sv = shap_vals_clase[:, k]
        fv = feat_vals_clase.iloc[:, k].values
        fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        colors = plt.cm.coolwarm(fv_norm)
        # Jitter vertical
        jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(sv))
        ax.scatter(sv, y_pos + jitter, c=colors, alpha=0.4, s=8, linewidths=0)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_names_top[::-1], fontsize=9)
    ax.set_xlabel("Valor SHAP", fontsize=10)
    ax.set_title(f"Clase: {clase}", fontsize=12, fontweight="bold",
                 color=palette_clase[clase])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

fig.suptitle("SHAP Beeswarm - Impacto de Features por Clase\n"
             "(Azul = valor bajo, Rojo = valor alto de la feature)",
             fontsize=14, fontweight="bold", y=1.02)

# Colorbar para la leyenda de valor de feature
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
cbar.set_label("Valor de la feature (normalizado)", fontsize=10)

fig.tight_layout()
save_fig(fig, "33_shap_beeswarm_por_clase.png")

# --- 4.3 SHAP Waterfall Plot: explicacion de una prediccion individual ---
print("[3/5] SHAP Waterfall Plot - Explicacion individual...")

# Seleccionar una muestra de cada clase para mostrar 3 explicaciones
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

y_shap_pred = rf_model.predict(X_shap)

for i, (clase_idx, clase_nombre) in enumerate(enumerate(clases)):
    ax = axes[i]
    # Buscar una muestra predicha correctamente como esta clase
    mask = y_shap_pred == clase_idx
    if mask.sum() == 0:
        ax.text(0.5, 0.5, f"Sin muestras\npredichas como\n{clase_nombre}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(f"Clase: {clase_nombre}", fontsize=12)
        continue

    sample_idx = np.where(mask)[0][0]
    sv = shap_values_rf[clase_idx][sample_idx]
    # expected_value puede ser escalar (binario) o array (multiclase)
    ev = explainer_rf.expected_value
    base_val = ev[clase_idx] if hasattr(ev, "__len__") else float(ev)

    # Ordenar por valor absoluto
    order = np.argsort(np.abs(sv))[::-1]
    top_k = min(8, len(sv))
    sv_top = sv[order[:top_k]]
    fn_top = [feature_cols[j] for j in order[:top_k]]
    cumsum = np.cumsum(sv_top)
    starts = np.concatenate([[base_val], base_val + cumsum[:-1]])

    colors_wf = ["#e74c3c" if v > 0 else "#3498db" for v in sv_top]

    ax.barh(range(top_k), sv_top, left=starts, color=colors_wf,
            edgecolor="black", linewidth=0.5, height=0.6)
    ax.axvline(x=base_val, color="gray", linewidth=1, linestyle="--", alpha=0.7, label=f"Base: {base_val:.3f}")
    ax.axvline(x=base_val + cumsum[-1], color="black", linewidth=1.5, linestyle="-",
               label=f"Pred: {base_val + cumsum[-1]:.3f}")

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(fn_top, fontsize=9)
    ax.set_xlabel("Contribucion SHAP", fontsize=10)
    ax.set_title(f"Prediccion: {clase_nombre}\n(Rojo = aumenta prob, Azul = reduce prob)",
                 fontsize=11, fontweight="bold", color=palette_clase[clase_nombre])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

fig.suptitle("SHAP Waterfall - Explicacion de Predicciones Individuales\n(Top 8 features por clase)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "34_shap_waterfall_individual.png")

# --- 4.4 SHAP Dependence Plot: UtilidadNeta vs SHAP ---
print("[4/5] SHAP Dependence Plot - UtilidadNeta...")

# Feature mas importante segun SHAP global
top_feature = shap_importance.iloc[-1]["Feature"]
top_feature_idx = feature_cols.index(top_feature)

# Segunda feature mas importante (para colorear puntos)
second_feature = shap_importance.iloc[-2]["Feature"]
second_feature_idx = feature_cols.index(second_feature)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, (clase_nombre, ax) in enumerate(zip(clases, axes)):
    sv_feat = shap_values_rf[i][:, top_feature_idx]
    feat_vals = X_shap.iloc[:, top_feature_idx].values
    color_vals = X_shap.iloc[:, second_feature_idx].values

    sc = ax.scatter(feat_vals, sv_feat, c=color_vals, cmap="viridis",
                    alpha=0.5, s=15, linewidths=0)
    plt.colorbar(sc, ax=ax, label=second_feature, shrink=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")

    # Linea de tendencia
    z = np.polyfit(feat_vals, sv_feat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(feat_vals.min(), feat_vals.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1.5, alpha=0.8, label="Tendencia")

    ax.set_xlabel(f"{top_feature} (escalado)", fontsize=10)
    ax.set_ylabel(f"SHAP value ({clase_nombre})", fontsize=10)
    ax.set_title(f"Clase: {clase_nombre}", fontsize=12, fontweight="bold",
                 color=palette_clase[clase_nombre])
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)

fig.suptitle(f"SHAP Dependence Plot: {top_feature}\n"
             f"(Coloreado por {second_feature})",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "35_shap_dependence_plot.png")

# --- 4.5 SHAP Heatmap: resumen compacto por clase ---
print("[5/5] SHAP Heatmap - Resumen de importancias por clase...")

# Importancia SHAP media absoluta por clase y feature
shap_por_clase = pd.DataFrame(
    {clase: np.abs(shap_values_rf[i]).mean(axis=0) for i, clase in enumerate(clases)},
    index=feature_cols
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(shap_por_clase, annot=True, fmt=".4f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "SHAP media absoluta"})
ax.set_title("SHAP - Importancia Media Absoluta por Feature y Clase\n(Random Forest)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Clase de Desempeno", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
fig.tight_layout()
save_fig(fig, "36_shap_heatmap_por_clase.png")

print("\n  SHAP completado. SHAP values calculados sobre muestra de "
      f"{SHAP_SAMPLE:,} registros del conjunto de prueba.")

# =========================================================
# 5. TECNICA 2: PERMUTATION FEATURE IMPORTANCE
# =========================================================
print("\n" + "=" * 60)
print("  TECNICA 2: PERMUTATION FEATURE IMPORTANCE")
print("  Modelos: Arbol de Decision, SVM, Random Forest")
print("=" * 60)

modelos_pfi = {
    "Arbol de Decision": dt_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
}

colores_modelo = {
    "Arbol de Decision": "#3498db",
    "SVM": "#e67e22",
    "Random Forest": "#27ae60",
}

# Muestra para PFI (SVM es lento)
PFI_SAMPLE = min(5000, len(X_test))
X_pfi = X_test.sample(n=PFI_SAMPLE, random_state=42)
y_pfi = y_test.loc[X_pfi.index]

print(f"\n  Muestra para PFI: {PFI_SAMPLE:,} registros")
print(f"  Repeticiones: 5 (para calcular media y desviacion estandar)")

resultados_pfi = {}
for nombre, modelo in modelos_pfi.items():
    print(f"\n  Calculando PFI para {nombre}...")
    pfi = permutation_importance(
        modelo, X_pfi, y_pfi,
        n_repeats=5,
        random_state=42,
        scoring="f1_weighted",
        n_jobs=-1
    )
    resultados_pfi[nombre] = {
        "mean": pfi.importances_mean,
        "std": pfi.importances_std,
        "raw": pfi.importances,
    }
    # Top 3 features
    top3_idx = np.argsort(pfi.importances_mean)[::-1][:3]
    print(f"    Top 3 features: " +
          ", ".join([f"{feature_cols[i]} ({pfi.importances_mean[i]:.4f})"
                     for i in top3_idx]))

# --- 5.1 Barplot comparativo de PFI ---
print("\n[1/3] Barplot comparativo de PFI por modelo...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, (nombre, res) in zip(axes, resultados_pfi.items()):
    orden = np.argsort(res["mean"])
    feat_sorted = [feature_cols[i] for i in orden]
    means_sorted = res["mean"][orden]
    stds_sorted = res["std"][orden]

    colors_pfi = [colores_modelo[nombre] if m > 0 else "#bdc3c7"
                  for m in means_sorted]

    bars = ax.barh(feat_sorted, means_sorted, xerr=stds_sorted,
                   color=colors_pfi, edgecolor="black", linewidth=0.4,
                   error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"},
                   height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Disminucion en F1-Score (weighted)", fontsize=10)
    ax.set_title(nombre, fontsize=12, fontweight="bold", color=colores_modelo[nombre])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val, std in zip(bars, means_sorted, stds_sorted):
        if val > 0:
            ax.text(val + std + max(means_sorted) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

fig.suptitle("Permutation Feature Importance por Modelo\n"
             "(Barras = media ± std sobre 5 repeticiones | Valores negativos = feature no aporta)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "37_pfi_barplot_por_modelo.png")

# --- 5.2 Heatmap comparativo de PFI ---
print("[2/3] Heatmap comparativo de PFI (3 modelos)...")

pfi_df = pd.DataFrame(
    {nombre: res["mean"] for nombre, res in resultados_pfi.items()},
    index=feature_cols
)
# Ordenar por importancia en RF
pfi_df = pfi_df.sort_values("Random Forest", ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pfi_df, annot=True, fmt=".4f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Disminucion media en F1-Score"})
ax.set_title("Permutation Feature Importance - Comparativa de Modelos\n"
             "(Verde = feature importante | Rojo = feature perjudicial)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Modelo", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
fig.tight_layout()
save_fig(fig, "38_pfi_heatmap_comparativo.png")

# --- 5.3 Grafico de consistencia: SHAP vs PFI para Random Forest ---
print("[3/3] Comparativa SHAP vs PFI para Random Forest...")

shap_imp_rf = pd.Series(
    np.abs(shap_values_rf[clases.index("Alto")]).mean(axis=0) +
    np.abs(shap_values_rf[clases.index("Bajo")]).mean(axis=0) +
    np.abs(shap_values_rf[clases.index("Medio")]).mean(axis=0),
    index=feature_cols
)
pfi_imp_rf = pd.Series(resultados_pfi["Random Forest"]["mean"], index=feature_cols)

# Normalizar a [0,1] para comparar en la misma escala
shap_norm = (shap_imp_rf - shap_imp_rf.min()) / (shap_imp_rf.max() - shap_imp_rf.min())
pfi_norm = (pfi_imp_rf - pfi_imp_rf.min()) / (pfi_imp_rf.max() - pfi_imp_rf.min())

comp_df = pd.DataFrame({
    "SHAP (normalizado)": shap_norm,
    "PFI (normalizado)": pfi_norm,
}).sort_values("SHAP (normalizado)", ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(comp_df))
width = 0.35

bars1 = ax.bar(x - width / 2, comp_df["SHAP (normalizado)"], width,
               label="SHAP", color="#9b59b6", edgecolor="black", linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width / 2, comp_df["PFI (normalizado)"], width,
               label="Permutation Feature Importance", color="#27ae60",
               edgecolor="black", linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(comp_df.index, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Importancia normalizada [0, 1]", fontsize=11)
ax.set_title("Comparativa SHAP vs Permutation Feature Importance\n"
             "(Random Forest | Ambas tecnicas normalizadas a [0,1])",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, 1.2)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, "39_shap_vs_pfi_comparativa.png")

# =========================================================
# 6. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN DE EXPLICABILIDAD XAI")
print("=" * 60)

print("\n  TECNICA 1 - SHAP (Random Forest):")
print(f"    Muestra analizada: {SHAP_SAMPLE:,} registros")
print(f"    Feature mas importante (SHAP global): {shap_importance.iloc[-1]['Feature']}")
print(f"      SHAP medio absoluto: {shap_importance.iloc[-1]['SHAP_mean_abs']:.4f}")
print(f"    Feature menos importante: {shap_importance.iloc[0]['Feature']}")
print(f"      SHAP medio absoluto: {shap_importance.iloc[0]['SHAP_mean_abs']:.4f}")

print("\n  TECNICA 2 - Permutation Feature Importance (3 modelos):")
for nombre, res in resultados_pfi.items():
    top_idx = np.argmax(res["mean"])
    print(f"    {nombre}: feature mas importante = {feature_cols[top_idx]} "
          f"({res['mean'][top_idx]:.4f} ± {res['std'][top_idx]:.4f})")

print("\n  Graficos generados:")
graficos = [
    "32_shap_barplot_global.png",
    "33_shap_beeswarm_por_clase.png",
    "34_shap_waterfall_individual.png",
    "35_shap_dependence_plot.png",
    "36_shap_heatmap_por_clase.png",
    "37_pfi_barplot_por_modelo.png",
    "38_pfi_heatmap_comparativo.png",
    "39_shap_vs_pfi_comparativa.png",
]
for g in graficos:
    print(f"    - {g}")

print(f"\n  Imagenes en: {RESULTS_DIR}")
print("=" * 60)
print("\nExplicabilidad SHAP y Permutation Feature Importance completada.")
