# Vamos a remodelar la info para que me quede lista para el analisis.

import numpy as np
import pandas as pd
import seaborn as sns

path = "E:/Doctorado/protocol2023/" 
#path = "/media/pedro/Expansion/Doctorado/Protocol2023/"
file = "EEG_features_subject_level.xlsx"
filepath = path+file
data = pd.read_excel(filepath, sheet_name="subject_level")

id_cols = ["subject", "condition"]
band_col = "band"

shared_cols = [c for c in data.columns if c.startswith(("spec_exp_", "spec_off_", "te_","lzc"))]
feature_cols = [c for c in data.columns if c not in id_cols + [band_col]]
banded_cols = [c for c in feature_cols if c not in shared_cols]

for c in shared_cols:
    if (data.groupby(id_cols)[c].nunique(dropna=False) > 1).any():
        raise ValueError(f"La columna {c} no es constante dentro de cada subject-condition")

shared_df = (
    data[id_cols + shared_cols]
    .drop_duplicates(subset=id_cols, keep="first")
    .set_index(id_cols)
)

wide_banded = (
    data[id_cols + [band_col] + banded_cols]
    .pivot_table(index=id_cols, columns=band_col, values=banded_cols, aggfunc="first")
)

wide_banded.columns = [f"{feat}__{band}" for feat, band in wide_banded.columns]

df_wide = pd.concat([shared_df, wide_banded], axis=1).reset_index()
df_wide.to_excel("EEG_features_wide.xlsx", index=False)