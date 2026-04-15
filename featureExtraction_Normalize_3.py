import pandas as pd
from sklearn.preprocessing import StandardScaler

input_file = "EEG_features_wide.xlsx"
output_file = "EEG_features_norm.xlsx"

id_cols = ["subject", "condition"]
df = pd.read_excel(input_file)

feature_cols = [c for c in df.columns if c not in id_cols]
numeric_feature_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

scaler = StandardScaler()
df_norm = df.copy()
df_norm[numeric_feature_cols] = scaler.fit_transform(df[numeric_feature_cols])

df_norm.to_excel(output_file, index=False)
print(f"Archivo cargado: {input_file}")
print(f"Filas y columnas: {df.shape}")
print(f"Cantidad de variables normalizadas: {len(numeric_feature_cols)}")
print(f"Archivo exportado: {output_file}")

print(df.describe())
print(df_norm.describe())

