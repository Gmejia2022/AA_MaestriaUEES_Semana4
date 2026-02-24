# -*- coding: utf-8 -*-
"""
9_VisualizacionesXAI_Integradas.py - Visualizaciones integradas de Explicabilidad XAI

Materia: APRENDIZAJE AUTOMATICO
Universidad de Especialidades Espiritu Santo (UEES)
Maestria en Inteligencia Artificial

Estudiantes:
  - Ingeniero Gonzalo Mejia Alcivar
  - Ingeniero Jorge Ortiz Merchan
  - Ingeniero David Perugachi Rojas

Objetivo:
  Generar visualizaciones integradas que cumplan con los 3 requerimientos XAI:

  1. Variables con mayor impacto:
     - Panel consolidado: SHAP global + PFI + Gini RF + Gini DT (4 tecnicas)
     - Ranking de features con consenso entre tecnicas
     - Radar de importancia normalizada por feature

  2. Comparacion de explicaciones entre tecnicas:
     - Tabla heatmap: SHAP vs PFI vs Gini DT vs Gini RF vs PDP
     - Dispersion SHAP vs PFI (concordancia entre metodos)
     - PDP de la feature top para los 3 modelos en un solo panel

  3. Casos individuales con decisiones explicadas:
     - Empresa 1: perfil financiero "Alto" - waterfall SHAP + reglas del arbol
     - Empresa 2: perfil financiero "Bajo" - waterfall SHAP + reglas del arbol
     - Panel resumen: confianza por clase para ambas empresas

  Prerequisito: ejecutar 7_ExplicabilidadSHAP.py y 8_ExplicabilidadPDP_Arbol.py primero
  (o tener los modelos .pkl en Models/ y el dataset en Data/)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.max_open_warning": 0})

# =========================================================
# 1. CONFIGURACION
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "DataSet2024.csv")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_START = 48  # Las imagenes de este script empiezan en 48


def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Guardado: {path}")


# =========================================================
# 2. CARGA DE MODELOS Y DATOS
# =========================================================
print("=" * 60)
print("  VISUALIZACIONES XAI INTEGRADAS")
print("  Variables de impacto | Comparacion | Casos individuales")
print("=" * 60)

print("\n  Cargando modelos y artefactos...")
dt_model  = joblib.load(os.path.join(MODELS_DIR, "arbol_decision.pkl"))
rf_model  = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
svm_model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_target = joblib.load(os.path.join(MODELS_DIR, "label_encoder_target.pkl"))
le_sector = joblib.load(os.path.join(MODELS_DIR, "label_encoder_sector.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

clases = list(le_target.classes_)  # ['Alto', 'Bajo', 'Medio']
n_feat = len(feature_cols)

# =========================================================
# 3. RECONSTRUCCION DEL CONJUNTO DE PRUEBA
# =========================================================
print("\n  Reconstruyendo datos...")
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

SAMPLE = min(2000, len(X_test))
X_sample = X_test.sample(n=SAMPLE, random_state=42)
y_sample = y_test.loc[X_sample.index]

print(f"  Prueba: {X_test.shape[0]:,} | Muestra XAI: {SAMPLE:,}")

# =========================================================
# 4. CALCULAR IMPORTANCIAS (todas las tecnicas)
# =========================================================
print("\n  Calculando importancias de features (4 tecnicas)...")

# --- Gini Impurity ---
imp_gini_dt = pd.Series(dt_model.feature_importances_, index=feature_cols)
imp_gini_rf = pd.Series(rf_model.feature_importances_, index=feature_cols)

# --- Permutation Feature Importance (RF, muestra reducida) ---
print("  PFI (Random Forest)...")
pfi_result = permutation_importance(
    rf_model, X_sample, y_sample,
    n_repeats=5, random_state=42, scoring="f1_weighted", n_jobs=-1
)
imp_pfi = pd.Series(pfi_result.importances_mean, index=feature_cols)
imp_pfi_std = pd.Series(pfi_result.importances_std, index=feature_cols)

# --- SHAP (TreeExplainer sobre RF) ---
print("  SHAP (Random Forest)...")
explainer = shap.TreeExplainer(rf_model)
shap_raw = explainer.shap_values(X_sample)

# Normalizar estructura de SHAP segun version
if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
    shap_vals = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
elif isinstance(shap_raw, list):
    shap_vals = shap_raw
else:
    shap_vals = [shap_raw]

# Importancia SHAP global: media absoluta promediando todas las clases
imp_shap = pd.Series(
    np.mean([np.abs(shap_vals[i]).mean(axis=0) for i in range(len(clases))], axis=0),
    index=feature_cols
)

# --- PDP range (feature top4, clase Alto) ---
print("  PDP range (Random Forest)...")
idx_alto = clases.index("Alto")
feat_imp_order = np.argsort(imp_gini_rf.values)[::-1]
top4_idx = feat_imp_order[:4].tolist()
top4_features = [feature_cols[i] for i in top4_idx]

imp_pdp = {}
for fi, fn in zip(top4_idx, top4_features):
    try:
        pd_res = partial_dependence(rf_model, X_sample, features=[fi],
                                    kind="average", grid_resolution=30)
        avg_r = pd_res["average"]
        avg_alto = avg_r[idx_alto] if avg_r.ndim == 2 else avg_r[0]
        imp_pdp[fn] = float(avg_alto.max() - avg_alto.min())
    except Exception:
        imp_pdp[fn] = 0.0
for f in feature_cols:
    if f not in imp_pdp:
        imp_pdp[f] = 0.0
imp_pdp_s = pd.Series(imp_pdp)

# Normalizar todas a [0,1]
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

n_shap    = norm01(imp_shap)
n_pfi     = norm01(imp_pfi.clip(lower=0))
n_gini_rf = norm01(imp_gini_rf)
n_gini_dt = norm01(imp_gini_dt)
n_pdp     = norm01(imp_pdp_s)

# Orden por SHAP
order_shap = n_shap.sort_values(ascending=False).index.tolist()

palette_clase = {"Alto": "#27ae60", "Bajo": "#e74c3c", "Medio": "#f39c12"}
colores_modelo = {
    "Arbol de Decision": "#3498db",
    "SVM": "#e67e22",
    "Random Forest": "#27ae60",
}
TECNICAS_COLORS = {
    "SHAP": "#9b59b6",
    "PFI": "#27ae60",
    "Gini RF": "#3498db",
    "Gini DT": "#e67e22",
    "PDP Range": "#e74c3c",
}

print(f"\n  Top 3 features (SHAP): {order_shap[:3]}")
print(f"  Top 3 features (Gini RF): {imp_gini_rf.sort_values(ascending=False).index[:3].tolist()}")
print(f"  Top 3 features (PFI): {imp_pfi.sort_values(ascending=False).index[:3].tolist()}")

# =========================================================
# 5. BLOQUE 1: VARIABLES CON MAYOR IMPACTO
# =========================================================
print("\n" + "=" * 60)
print("  BLOQUE 1: VARIABLES CON MAYOR IMPACTO")
print("=" * 60)

# --- 48: Panel consolidado de 4 tecnicas (barras agrupadas) ---
print("\n[1/3] Panel consolidado de importancias (4 tecnicas)...")

comp_df = pd.DataFrame({
    "SHAP": n_shap,
    "PFI": n_pfi,
    "Gini RF": n_gini_rf,
    "Gini DT": n_gini_dt,
}, index=feature_cols).loc[order_shap]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(comp_df))
width = 0.20
offsets = [-1.5, -0.5, 0.5, 1.5]
cols = ["SHAP", "PFI", "Gini RF", "Gini DT"]
color_list = [TECNICAS_COLORS[c] for c in cols]

for off, col, clr in zip(offsets, cols, color_list):
    bars = ax.bar(x + off * width, comp_df[col], width,
                  label=col, color=clr, edgecolor="black", linewidth=0.4, alpha=0.88)

# Anotar la feature mas importante con flechas
top_feat = order_shap[0]
top_x = 0
ax.annotate(f"Feature mas\nimportante:\n{top_feat}",
            xy=(top_x - 1.5 * width, comp_df.loc[top_feat, "SHAP"]),
            xytext=(top_x + 2.5, 0.9),
            fontsize=9, color="#9b59b6", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(comp_df.index, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Importancia normalizada [0, 1]", fontsize=11)
ax.set_title("Variables con Mayor Impacto en el Modelo\n"
             "(4 tecnicas XAI | todas normalizadas a [0,1] | Random Forest)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 1.25)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
save_fig(fig, f"{IMG_START}_impacto_panel_consolidado.png")

# --- 49: Ranking con consenso entre tecnicas ---
print("[2/3] Ranking de consenso entre tecnicas...")

# Score de consenso: promedio de las 4 importancias normalizadas
consenso = pd.DataFrame({
    "SHAP": n_shap,
    "PFI": n_pfi,
    "Gini RF": n_gini_rf,
    "Gini DT": n_gini_dt,
}).mean(axis=1).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Izquierda: barras de consenso con error (std entre tecnicas)
std_consenso = pd.DataFrame({
    "SHAP": n_shap, "PFI": n_pfi, "Gini RF": n_gini_rf, "Gini DT": n_gini_dt,
}).std(axis=1).loc[consenso.index]

colors_cons = plt.cm.RdYlGn(consenso.values / consenso.max())
bars = axes[0].barh(consenso.index, consenso.values, xerr=std_consenso.values,
                    color=colors_cons, edgecolor="black", linewidth=0.4,
                    error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"},
                    height=0.6)
for bar, val in zip(bars, consenso.values):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
axes[0].axvline(x=consenso.mean(), color="red", linewidth=1.2, linestyle="--",
                label=f"Media: {consenso.mean():.3f}")
axes[0].set_xlabel("Score de consenso (promedio normalizado)", fontsize=10)
axes[0].set_title("Ranking de Features por Consenso\n(Promedio de 4 tecnicas XAI)",
                  fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="x", linestyle="--", alpha=0.4)

# Derecha: radar de importancia por feature (top 6)
top6 = consenso.sort_values(ascending=False).index[:6].tolist()
radar_df = pd.DataFrame({
    "SHAP": n_shap[top6],
    "PFI": n_pfi[top6],
    "Gini RF": n_gini_rf[top6],
    "Gini DT": n_gini_dt[top6],
}, index=top6)

x6 = np.arange(len(top6))
w6 = 0.18
off6 = [-1.5, -0.5, 0.5, 1.5]
for off, col, clr in zip(off6, ["SHAP", "PFI", "Gini RF", "Gini DT"],
                          [TECNICAS_COLORS[c] for c in ["SHAP", "PFI", "Gini RF", "Gini DT"]]):
    axes[1].bar(x6 + off * w6, radar_df[col], w6,
                label=col, color=clr, edgecolor="black", linewidth=0.4, alpha=0.85)

axes[1].set_xticks(x6)
axes[1].set_xticklabels(top6, rotation=25, ha="right", fontsize=9)
axes[1].set_ylabel("Importancia normalizada", fontsize=10)
axes[1].set_title("Top 6 Features - Detalle por Tecnica",
                  fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 1.2)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Analisis de Consenso: Variables con Mayor Impacto en las Decisiones del Modelo",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, f"{IMG_START + 1}_impacto_ranking_consenso.png")

# --- 50: Heatmap de importancias (todas las tecnicas x todas las features) ---
print("[3/3] Heatmap de impacto consolidado...")

heatmap_df = pd.DataFrame({
    "SHAP": n_shap,
    "PFI": n_pfi,
    "Gini RF": n_gini_rf,
    "Gini DT": n_gini_dt,
    "PDP Range": n_pdp,
}, index=feature_cols).loc[order_shap]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Importancia normalizada [0, 1]"})
ax.set_title("Mapa de Calor de Importancia: 5 Tecnicas XAI x 10 Features\n"
             "(Ordenado por SHAP | Mayor valor = mayor impacto en la decision)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Tecnica XAI", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
fig.tight_layout()
save_fig(fig, f"{IMG_START + 2}_impacto_heatmap_5tecnicas.png")

# =========================================================
# 6. BLOQUE 2: COMPARACION DE EXPLICACIONES
# =========================================================
print("\n" + "=" * 60)
print("  BLOQUE 2: COMPARACION DE EXPLICACIONES ENTRE TECNICAS")
print("=" * 60)

# --- 51: Dispersion SHAP vs PFI (concordancia) ---
print("\n[1/3] Dispersion SHAP vs PFI (concordancia)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pares = [("SHAP", "PFI", n_shap, n_pfi),
         ("SHAP", "Gini RF", n_shap, n_gini_rf),
         ("PFI", "Gini RF", n_pfi, n_gini_rf)]

for ax, (lbl_x, lbl_y, sx, sy) in zip(axes, pares):
    ax.scatter(sx.values, sy.values, s=120, c=range(n_feat),
               cmap="tab10", edgecolors="black", linewidths=0.6, zorder=3)
    # Etiquetas de features
    for feat, vx, vy in zip(feature_cols, sx.values, sy.values):
        ax.annotate(feat, (vx, vy), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color="#2c3e50")
    # Linea diagonal de concordancia perfecta
    lim = max(sx.max(), sy.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, alpha=0.7, label="Concordancia perfecta")
    # Correlacion de Spearman
    corr = pd.Series(sx.values).corr(pd.Series(sy.values), method="spearman")
    ax.set_xlabel(f"{lbl_x} (normalizado)", fontsize=10)
    ax.set_ylabel(f"{lbl_y} (normalizado)", fontsize=10)
    ax.set_title(f"{lbl_x} vs {lbl_y}\nr_spearman = {corr:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)

fig.suptitle("Concordancia entre Tecnicas XAI\n"
             "(Cada punto es una feature | Linea roja = concordancia perfecta)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, f"{IMG_START + 3}_comparacion_concordancia.png")

# --- 52: PDP de la feature top para los 3 modelos (panel unico) ---
print("[2/3] PDP comparativo de feature top para los 3 modelos...")

top1_feat = order_shap[0]
top1_idx_val = feature_cols.index(top1_feat)
modelos_pdp = {"Arbol de Decision": dt_model, "SVM": svm_model, "Random Forest": rf_model}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, clase_nombre in zip(axes, clases):
    clase_idx = clases.index(clase_nombre)
    for nombre_mod, modelo in modelos_pdp.items():
        try:
            pd_res = partial_dependence(modelo, X_sample, features=[top1_idx_val],
                                        kind="average", grid_resolution=40)
            gv = pd_res["grid_values"][0]
            avg_r = pd_res["average"]
            avg_p = avg_r[clase_idx] if avg_r.ndim == 2 else avg_r[0]
            ax.plot(gv, avg_p, color=colores_modelo[nombre_mod],
                    linewidth=2.5, label=nombre_mod)
            ax.fill_between(gv, avg_p, alpha=0.07, color=colores_modelo[nombre_mod])
        except Exception:
            pass
    ax.set_xlabel(f"{top1_feat} (escalado)", fontsize=10)
    ax.set_ylabel("Probabilidad predicha" if ax == axes[0] else "", fontsize=10)
    ax.set_title(f"Clase: {clase_nombre}", fontsize=12, fontweight="bold",
                 color=palette_clase[clase_nombre])
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

fig.suptitle(f"Comparacion PDP: {top1_feat} (Feature #1 segun SHAP)\n"
             "Como cambia la prediccion de cada clase al variar la variable mas importante\n"
             "segun los 3 modelos entrenados",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, f"{IMG_START + 4}_comparacion_pdp_modelos.png")

# --- 53: Tabla resumen comparativa de tecnicas XAI ---
print("[3/3] Tabla resumen de comparacion de tecnicas...")

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Izquierda: tabla de caracteristicas de cada tecnica
ax_tabla = fig.add_subplot(gs[0, 0])
ax_tabla.axis("off")

filas = [
    ["SHAP", "Local + Global", "RF (TreeExplainer)", "Alta", "Alta"],
    ["PFI", "Global", "3 modelos", "Media", "Alta"],
    ["Gini RF", "Global", "Random Forest", "Media", "Media"],
    ["Gini DT", "Global", "Arbol Decision", "Alta", "Media"],
    ["PDP", "Global marginal", "RF (predict_proba)", "Alta", "Media"],
]
cols_tabla = ["Tecnica", "Tipo", "Modelo", "Interpretabilidad", "Consistencia"]

tabla = ax_tabla.table(
    cellText=filas,
    colLabels=cols_tabla,
    cellLoc="center",
    loc="center",
    colWidths=[0.18, 0.22, 0.22, 0.2, 0.2]
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1, 2.0)

for j in range(len(cols_tabla)):
    tabla[0, j].set_facecolor("#2c3e50")
    tabla[0, j].set_text_props(color="white", fontweight="bold")

fila_colors = ["#d5e8f5", "#d5f0e8", "#fde8d5", "#f5d5d5", "#ede8f5"]
for i, clr in enumerate(fila_colors):
    for j in range(len(cols_tabla)):
        tabla[i + 1, j].set_facecolor(clr)

ax_tabla.set_title("Caracteristicas de cada Tecnica XAI",
                   fontsize=12, fontweight="bold", pad=20)

# Derecha: ranking consolidado (barplot horizontal)
ax_rank = fig.add_subplot(gs[0, 1])
ranking_final = pd.DataFrame({
    "SHAP": n_shap,
    "PFI": n_pfi,
    "Gini RF": n_gini_rf,
    "Gini DT": n_gini_dt,
    "PDP": n_pdp,
}, index=feature_cols).mean(axis=1).sort_values(ascending=True)

colors_rank = plt.cm.YlOrRd(ranking_final.values / ranking_final.max())
bars = ax_rank.barh(ranking_final.index, ranking_final.values,
                    color=colors_rank, edgecolor="black", linewidth=0.5, height=0.6)
for bar, val in zip(bars, ranking_final.values):
    ax_rank.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

ax_rank.set_xlabel("Score de consenso (5 tecnicas)", fontsize=10)
ax_rank.set_title("Ranking Final de Features\n(Promedio de 5 tecnicas XAI)",
                  fontsize=12, fontweight="bold")
ax_rank.grid(axis="x", linestyle="--", alpha=0.4)
ax_rank.set_xlim(0, 1.15)

fig.suptitle("Resumen Comparativo de Tecnicas XAI",
             fontsize=15, fontweight="bold", y=1.01)
save_fig(fig, f"{IMG_START + 5}_comparacion_tabla_resumen.png")

# =========================================================
# 7. BLOQUE 3: CASOS INDIVIDUALES CON DECISIONES EXPLICADAS
# =========================================================
print("\n" + "=" * 60)
print("  BLOQUE 3: CASOS INDIVIDUALES CON DECISIONES EXPLICADAS")
print("=" * 60)

# Seleccionar 2 ejemplos concretos: uno Alto y uno Bajo (prediccion correcta)
y_pred_all = rf_model.predict(X_sample)

# Caso 1: empresa clasificada como Alto
mask_alto = (y_pred_all == clases.index("Alto")) & (y_sample.values == clases.index("Alto"))
idx_caso1 = X_sample.index[mask_alto][0]
empresa1 = X_sample.loc[idx_caso1]
shap_caso1 = shap_vals[clases.index("Alto")][np.where(X_sample.index == idx_caso1)[0][0]]
prob_caso1 = rf_model.predict_proba(empresa1.values.reshape(1, -1))[0]

# Caso 2: empresa clasificada como Bajo
mask_bajo = (y_pred_all == clases.index("Bajo")) & (y_sample.values == clases.index("Bajo"))
idx_caso2 = X_sample.index[mask_bajo][0]
empresa2 = X_sample.loc[idx_caso2]
shap_caso2 = shap_vals[clases.index("Bajo")][np.where(X_sample.index == idx_caso2)[0][0]]
prob_caso2 = rf_model.predict_proba(empresa2.values.reshape(1, -1))[0]

ev = explainer.expected_value
base_alto = ev[clases.index("Alto")] if hasattr(ev, "__len__") else float(ev)
base_bajo  = ev[clases.index("Bajo")]  if hasattr(ev, "__len__") else float(ev)

print(f"\n  Caso 1 - Empresa 'Alto': idx={idx_caso1} | prob={prob_caso1}")
print(f"  Caso 2 - Empresa 'Bajo': idx={idx_caso2} | prob={prob_caso2}")

# --- 54: Waterfall SHAP para Caso 1 (Alto) ---
print("\n[1/3] Waterfall SHAP - Caso 1 (empresa Alto)...")


def waterfall_panel(ax, sv, base_val, feat_names, titulo, color_pos, color_neg, prob_clases):
    order = np.argsort(np.abs(sv))[::-1]
    top_k = min(8, len(sv))
    sv_top = sv[order[:top_k]]
    fn_top = [feat_names[j] for j in order[:top_k]]
    cumsum = np.cumsum(sv_top)
    starts = np.concatenate([[base_val], base_val + cumsum[:-1]])
    colors_wf = [color_pos if v > 0 else color_neg for v in sv_top]

    bars = ax.barh(range(top_k), sv_top, left=starts, color=colors_wf,
                   edgecolor="black", linewidth=0.5, height=0.6)

    for j, (bar, val, feat) in enumerate(zip(bars, sv_top, fn_top)):
        sign = "+" if val > 0 else ""
        ax.text(starts[j] + val + (sv_top.max() - sv_top.min()) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.4f}", va="center", fontsize=8)

    pred_final = base_val + cumsum[-1]
    ax.axvline(x=base_val, color="gray", linewidth=1, linestyle="--", alpha=0.7,
               label=f"Valor base: {base_val:.3f}")
    ax.axvline(x=pred_final, color="black", linewidth=2, linestyle="-",
               label=f"Prediccion: {pred_final:.3f}")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(fn_top, fontsize=9)
    ax.set_xlabel("Contribucion SHAP", fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Mini-barplot de probabilidades en el margen
    ax2 = ax.inset_axes([1.05, 0, 0.35, 1])
    colores_prob = ["#27ae60", "#e74c3c", "#f39c12"]
    y_idx = np.arange(len(clases))
    ax2.barh(y_idx, prob_clases, color=colores_prob, edgecolor="black", linewidth=0.4)
    ax2.set_xlim(0, 1)
    ax2.set_yticks(y_idx)
    ax2.set_yticklabels(clases, fontsize=8)
    ax2.set_xlabel("Probabilidad", fontsize=8)
    ax2.set_title("Conf.\npor clase", fontsize=8, fontweight="bold")
    for j, (cls, p) in enumerate(zip(clases, prob_clases)):
        ax2.text(p + 0.02, j, f"{p:.2%}", va="center", fontsize=7.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(22, 8))
fig.subplots_adjust(wspace=0.6)

waterfall_panel(
    axes[0], shap_caso1, base_alto, feature_cols,
    "Caso 1: Empresa con Desempeno ALTO\n(Rojo = aumenta probabilidad Alto | Azul = reduce)",
    "#e74c3c", "#3498db", prob_caso1
)
waterfall_panel(
    axes[1], shap_caso2, base_bajo, feature_cols,
    "Caso 2: Empresa con Desempeno BAJO\n(Rojo = aumenta probabilidad Bajo | Azul = reduce)",
    "#e74c3c", "#3498db", prob_caso2
)

fig.suptitle("SHAP Waterfall - Explicacion de Predicciones Individuales\n"
             "Top 8 features que mas influyeron en la decision del modelo (Random Forest)",
             fontsize=14, fontweight="bold", y=1.02)
save_fig(fig, f"{IMG_START + 6}_casos_waterfall_shap.png")

# --- 55: Perfil financiero de los 2 casos (radar) ---
print("[2/3] Perfil financiero de los 2 casos (radar)...")

top6_feats = order_shap[:6]

fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                         subplot_kw=dict(polar=True))

casos = [
    (empresa1, "Empresa Alto", "#27ae60", prob_caso1),
    (empresa2, "Empresa Bajo", "#e74c3c", prob_caso2),
]

for ax, (empresa, label, color, probs) in zip(axes, casos):
    vals = [float(empresa[f]) for f in top6_feats]
    # Normalizar respecto al rango en X_sample
    vals_norm = []
    for f, v in zip(top6_feats, vals):
        mn = X_sample[f].min()
        mx = X_sample[f].max()
        vals_norm.append((v - mn) / (mx - mn + 1e-9))

    angles = np.linspace(0, 2 * np.pi, len(top6_feats), endpoint=False).tolist()
    vals_norm += vals_norm[:1]
    angles += angles[:1]

    ax.fill(angles, vals_norm, alpha=0.25, color=color)
    ax.plot(angles, vals_norm, "o-", linewidth=2, color=color, markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top6_feats, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"{label}\n"
                 f"P(Alto)={probs[clases.index('Alto')]:.1%} | "
                 f"P(Bajo)={probs[clases.index('Bajo')]:.1%} | "
                 f"P(Medio)={probs[clases.index('Medio')]:.1%}",
                 fontsize=11, fontweight="bold", color=color, pad=15)
    ax.grid(linestyle="--", alpha=0.5)

fig.suptitle("Perfil Financiero de los Casos Individuales\n"
             "(Top 6 features segun SHAP | Valores normalizados al rango del conjunto de prueba)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save_fig(fig, f"{IMG_START + 7}_casos_perfil_radar.png")

# --- 56: Panel completo: reglas del arbol + SHAP + confianza ---
print("[3/3] Panel resumen de casos individuales...")

fig = plt.figure(figsize=(22, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.45)

# Fila 0: Caso 1 (Alto) — waterfall simplificado + confianza + tabla de valores
# Fila 1: Caso 2 (Bajo) — waterfall simplificado + confianza + tabla de valores

for row_idx, (empresa, clase_pred, shap_sv, base_val, probs, clr) in enumerate([
    (empresa1, "Alto", shap_caso1, base_alto, prob_caso1, "#27ae60"),
    (empresa2, "Bajo", shap_caso2, base_bajo, prob_caso2, "#e74c3c"),
]):
    # --- Columna 0: Waterfall compacto ---
    ax_wf = fig.add_subplot(gs[row_idx, 0])
    order = np.argsort(np.abs(shap_sv))[::-1]
    top5 = min(5, len(shap_sv))
    sv5 = shap_sv[order[:top5]]
    fn5 = [feature_cols[j] for j in order[:top5]]
    cumsum5 = np.cumsum(sv5)
    starts5 = np.concatenate([[base_val], base_val + cumsum5[:-1]])
    cols_wf = ["#e74c3c" if v > 0 else "#3498db" for v in sv5]
    ax_wf.barh(range(top5), sv5, left=starts5, color=cols_wf,
               edgecolor="black", linewidth=0.5, height=0.6)
    ax_wf.axvline(x=base_val, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax_wf.axvline(x=base_val + cumsum5[-1], color="black", linewidth=1.5)
    ax_wf.set_yticks(range(top5))
    ax_wf.set_yticklabels(fn5, fontsize=9)
    ax_wf.set_title(f"Caso {'1' if row_idx == 0 else '2'}: {clase_pred}\nSHAP (top 5 features)",
                    fontsize=10, fontweight="bold", color=clr)
    ax_wf.grid(axis="x", linestyle="--", alpha=0.3)

    # --- Columna 1: Confianza del modelo (barplot vertical) ---
    ax_conf = fig.add_subplot(gs[row_idx, 1])
    colors_prob = ["#27ae60", "#e74c3c", "#f39c12"]
    bars_conf = ax_conf.bar(clases, probs, color=colors_prob,
                            edgecolor="black", linewidth=0.5)
    for bar, p in zip(bars_conf, probs):
        ax_conf.text(bar.get_x() + bar.get_width() / 2, p + 0.02,
                     f"{p:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax_conf.set_ylim(0, 1.15)
    ax_conf.set_ylabel("Probabilidad predicha", fontsize=9)
    ax_conf.set_title(f"Confianza del modelo\n(Prediccion: {clase_pred})",
                      fontsize=10, fontweight="bold", color=clr)
    ax_conf.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    # Resaltar la clase predicha
    idx_pred = clases.index(clase_pred)
    bars_conf[idx_pred].set_edgecolor("black")
    bars_conf[idx_pred].set_linewidth(2.5)
    ax_conf.grid(axis="y", linestyle="--", alpha=0.3)

    # --- Columna 2: Tabla de valores de la empresa ---
    ax_tbl = fig.add_subplot(gs[row_idx, 2])
    ax_tbl.axis("off")

    # Top 5 features por SHAP
    tbl_data = [[fn5[k], f"{empresa[fn5[k]]:.4f}", f"{sv5[k]:+.4f}"]
                for k in range(top5)]
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Feature", "Valor (escalado)", "SHAP"],
        cellLoc="center",
        loc="center",
        colWidths=[0.42, 0.30, 0.28]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for k in range(top5):
        bg = "#fde8e8" if sv5[k] > 0 else "#e8f0fd"
        for j in range(3):
            tbl[k + 1, j].set_facecolor(bg)
    ax_tbl.set_title(f"Valores de la Empresa\n(Rojo = empuja a {clase_pred} | Azul = reduce)",
                     fontsize=10, fontweight="bold", color=clr)

fig.suptitle("Panel de Explicacion de Casos Individuales\n"
             "Caso 1: Empresa de Desempeno ALTO  |  Caso 2: Empresa de Desempeno BAJO",
             fontsize=14, fontweight="bold", y=1.01)
save_fig(fig, f"{IMG_START + 8}_casos_panel_resumen.png")

# =========================================================
# 8. RESUMEN FINAL
# =========================================================
print("\n" + "=" * 60)
print("  RESUMEN - VISUALIZACIONES XAI INTEGRADAS")
print("=" * 60)

top3_consenso = ranking_final.sort_values(ascending=False).index[:3].tolist()
print(f"\n  Top 3 features por consenso (5 tecnicas): {top3_consenso}")
print(f"\n  Caso 1 (Empresa Alto):")
print(f"    P(Alto)  = {prob_caso1[clases.index('Alto')]:.2%}")
print(f"    P(Bajo)  = {prob_caso1[clases.index('Bajo')]:.2%}")
print(f"    P(Medio) = {prob_caso1[clases.index('Medio')]:.2%}")
print(f"    Feature mas decisiva: {feature_cols[np.argmax(np.abs(shap_caso1))]}")
print(f"\n  Caso 2 (Empresa Bajo):")
print(f"    P(Alto)  = {prob_caso2[clases.index('Alto')]:.2%}")
print(f"    P(Bajo)  = {prob_caso2[clases.index('Bajo')]:.2%}")
print(f"    P(Medio) = {prob_caso2[clases.index('Medio')]:.2%}")
print(f"    Feature mas decisiva: {feature_cols[np.argmax(np.abs(shap_caso2))]}")

graficos = [
    f"{IMG_START}_impacto_panel_consolidado.png",
    f"{IMG_START + 1}_impacto_ranking_consenso.png",
    f"{IMG_START + 2}_impacto_heatmap_5tecnicas.png",
    f"{IMG_START + 3}_comparacion_concordancia.png",
    f"{IMG_START + 4}_comparacion_pdp_modelos.png",
    f"{IMG_START + 5}_comparacion_tabla_resumen.png",
    f"{IMG_START + 6}_casos_waterfall_shap.png",
    f"{IMG_START + 7}_casos_perfil_radar.png",
    f"{IMG_START + 8}_casos_panel_resumen.png",
]
print("\n  Graficos generados:")
for g in graficos:
    print(f"    - {g}")

print(f"\n  Imagenes en: {RESULTS_DIR}")
print("=" * 60)
print("\nVisualizaciones XAI Integradas completadas.")
