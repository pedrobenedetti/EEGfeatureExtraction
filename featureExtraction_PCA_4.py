import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

input_file = "EEG_features_norm.xlsx"
output_excel = "EEG_PCA_results.xlsx"
output_txt = "EEG_PCA_summary.txt"

id_cols = ["subject", "condition"]
variance_threshold = 0.80

df = pd.read_excel(input_file)

feature_cols = [c for c in df.columns if c not in id_cols]
X = df[feature_cols].copy()
# chequeo de seguridad: todas deben ser numericas
non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
if len(non_numeric) > 0:
    raise ValueError(f"Hay columnas no numericas en X: {non_numeric}")

X_np = X.to_numpy()

pca = PCA()
scores = pca.fit_transform(X_np)

explained_variance_ratio = pca.explained_variance_ratio_
explained_variance = pca.explained_variance_
cumulative_variance = np.cumsum(explained_variance_ratio)

n_components_80 = np.argmax(cumulative_variance >= variance_threshold) + 1

# ============================================================================
# DATAFRAMES DE SALIDA
# ============================================================================

pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]

# scores
df_scores = pd.DataFrame(scores, columns=pc_names)
df_scores.insert(0, "condition", df["condition"])
df_scores.insert(0, "subject", df["subject"])

# loadings
# filas = variables originales, columnas = componentes
df_loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=pc_names
).reset_index().rename(columns={"index": "variable"})

# varianza explicada
df_variance = pd.DataFrame({
    "component": pc_names,
    "explained_variance": explained_variance,
    "explained_variance_ratio": explained_variance_ratio,
    "cumulative_variance_ratio": cumulative_variance
})

# ============================================================================
# EXPORT A EXCEL
# ============================================================================

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_scores.to_excel(writer, sheet_name="scores", index=False)
    df_loadings.to_excel(writer, sheet_name="loadings", index=False)
    df_variance.to_excel(writer, sheet_name="variance", index=False)

# ============================================================================
# RESUMEN EN TXT
# ============================================================================

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("PCA SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Archivo de entrada: {input_file}\n")
    f.write(f"Observaciones: {df.shape[0]}\n")
    f.write(f"Variables originales totales: {df.shape[1]}\n")
    f.write(f"Variables usadas en PCA: {len(feature_cols)}\n")
    f.write(f"Columnas excluidas: {id_cols}\n")
    f.write(f"Criterio de seleccion: {variance_threshold*100:.1f}% de varianza acumulada\n")
    f.write(f"Cantidad de componentes para alcanzar el criterio: {n_components_80}\n")
    f.write("\n")

    f.write("VARIANZA EXPLICADA POR COMPONENTE\n")
    f.write("-" * 60 + "\n")
    for i, (evr, cum) in enumerate(zip(explained_variance_ratio, cumulative_variance), start=1):
        f.write(
            f"PC{i}: var_exp={evr:.6f} | var_acum={cum:.6f}\n"
        )

# ============================================================================
# IMPRESION DE RESULTADOS
# ============================================================================

print("\n" + "=" * 60)
print("PCA COMPLETADO")
print("=" * 60)
print(f"Archivo de entrada: {input_file}")
print(f"Observaciones: {df.shape[0]}")
print(f"Variables usadas en PCA: {len(feature_cols)}")
print(f"Cantidad total de componentes: {len(pc_names)}")
print(f"Componentes necesarias para alcanzar 80% acumulado: {n_components_80}")
print("\nPrimeras 10 componentes:")
print(df_variance.head(10))

# ============================================================================
# GRAFICOS
# ============================================================================

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker="o")
plt.xlabel("Componente principal")
plt.ylabel("Proporcion de varianza explicada")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

# Varianza acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=variance_threshold, linestyle="--")
plt.axvline(x=n_components_80, linestyle="--")
plt.xlabel("Componente principal")
plt.ylabel("Varianza acumulada")
plt.title("Varianza Acumulada")
plt.grid(True)
plt.show()

# PC1 vs PC2
plt.figure(figsize=(10, 6))
for cond in sorted(df_scores["condition"].unique()):
    mask = df_scores["condition"] == cond
    plt.scatter(
        df_scores.loc[mask, "PC1"],
        df_scores.loc[mask, "PC2"],
        label=f"Cond {cond}"
    )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2")
plt.legend()
plt.grid(True)
plt.show()

# PC1 vs PC3
if scores.shape[1] >= 3:
    plt.figure(figsize=(10, 6))
    for cond in sorted(df_scores["condition"].unique()):
        mask = df_scores["condition"] == cond
        plt.scatter(
            df_scores.loc[mask, "PC1"],
            df_scores.loc[mask, "PC3"],
            label=f"Cond {cond}"
        )
    plt.xlabel("PC1")
    plt.ylabel("PC3")
    plt.title("PC1 vs PC3")
    plt.legend()
    plt.grid(True)
    plt.show()