# ============================================================================
# LIMPIEZA DEL ENTORNO Y DIRECTORIO DE TRABAJO
# ============================================================================
rm(list = ls())
cat("\014")
graphics.off()
setwd("/media/pedro/Expansion/Doctorado/protocol2023")

# ============================================================================
# LIBRERIAS
# ============================================================================
# readxl: leer archivos Excel
# dplyr: manipular tablas
# glmnet: ajustar modelos Ridge / Lasso / Elastic Net
# ggplot2: hacer graficos
# openxlsx: exportar resultados a Excel
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(glmnet)
  library(ggplot2)
  library(openxlsx)
})

# ============================================================================
# CONFIGURACION GENERAL
# ============================================================================
# Archivo de scores de PCA
pca_file <- "EEG_PCA_results.xlsx"

# Archivo de conducta
conduct_file <- "Protocolo2023_conducta.xlsx"

# Hoja del PCA donde estan los scores
pca_sheet <- "scores"

# Hoja del archivo conductual
conduct_sheet <- "Hoja1"

# Variable conductual a predecir
# Cambiar esta linea si queres usar otra variable
target_var <- "REY"

# Condicion que vamos a usar
selected_condition <- 100

# Sujetos a excluir
subjects_to_exclude <- c("02_test_2023", "15_test_2023")

# PCs que vamos a usar como predictores
pcs_to_use <- c("PC1", "PC2", "PC3")

# Grilla de alpha:
# alpha = 0 -> Ridge
# alpha = 1 -> Lasso
# entre 0 y 1 -> Elastic Net
alpha_grid <- c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)

# Carpeta para guardar graficos
plots_dir <- paste0("ElasticNet_condition_", selected_condition, "_plots")
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir)
}

# Archivos de salida
output_excel <- paste0("ElasticNet_condition_", selected_condition, "_", target_var, "_results.xlsx")
output_txt   <- paste0("ElasticNet_condition_", selected_condition, "_", target_var, "_summary.txt")

# ============================================================================
# LECTURA DE DATOS
# ============================================================================
# Leemos la hoja de scores del PCA
pca_scores <- read_excel(pca_file, sheet = pca_sheet)

# Leemos la tabla conductual
conduct <- read_excel(conduct_file, sheet = conduct_sheet)

# ============================================================================
# CHEQUEO DE COLUMNAS
# ============================================================================
cat("Columnas de PCA:\n")
print(colnames(pca_scores))

cat("\nColumnas de conducta:\n")
print(colnames(conduct))

# ============================================================================
# SELECCION DE LA CONDICION 100
# ============================================================================
# Nos quedamos solo con una fila por sujeto correspondiente a la condicion 100
# y con las tres PCs que vamos a usar
pca_cond100 <- pca_scores %>%
  filter(condition == selected_condition) %>%
  select(subject, condition, all_of(pcs_to_use)) %>%
  filter(!subject %in% subjects_to_exclude)

# ============================================================================
# PREPARACION DE LA TABLA CONDUCTUAL
# ============================================================================
# Nos quedamos con subject, Grupo y la variable target
conduct_small <- conduct %>%
  select(subject, Grupo, all_of(target_var)) %>%
  filter(!subject %in% subjects_to_exclude)

# ============================================================================
# MERGE ENTRE PCA Y CONDUCTA
# ============================================================================
# Unimos por subject para obtener una fila final por sujeto
data_model <- inner_join(pca_cond100, conduct_small, by = "subject")

# Renombramos la variable target a y para simplificar el codigo
data_model <- data_model %>%
  rename(y = all_of(target_var))

# Eliminamos filas con NA
data_model <- data_model %>%
  drop_na()

# Convertimos Grupo a factor
data_model <- data_model %>%
  mutate(Grupo = as.factor(Grupo))

# ============================================================================
# CHEQUEO DEL DATASET FINAL
# ============================================================================
cat("\n============================================================\n")
cat("DATASET FINAL PARA ELASTIC NET - CONDICION 100 + GRUPO\n")
cat("============================================================\n")
cat("Variable target:", target_var, "\n")
cat("Condicion usada:", selected_condition, "\n")
cat("Cantidad de sujetos:", nrow(data_model), "\n")
cat("Sujetos usados:\n")
print(data_model$subject)

cat("\nPrimeras filas:\n")
print(head(data_model))

cat("\nEstructura:\n")
print(str(data_model))

# ============================================================================
# MATRIZ DE DISEÑO
# ============================================================================
# Definimos la formula de predictores:
# Grupo + PC1 + PC2 + PC3 para la condicion 100
predictor_names <- c("Grupo", pcs_to_use)

formula_x <- as.formula(
  paste("~", paste(predictor_names, collapse = " + "))
)

# model.matrix transforma Grupo en dummy y prepara la matriz para glmnet
x_full <- model.matrix(formula_x, data = data_model)[, -1, drop = FALSE]
y_full <- data_model$y

# ============================================================================
# FUNCIONES AUXILIARES PARA METRICAS
# ============================================================================
rmse_fun <- function(obs, pred) {
  sqrt(mean((obs - pred)^2))
}

mae_fun <- function(obs, pred) {
  mean(abs(obs - pred))
}

r2_fun <- function(obs, pred) {
  1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
}

# ============================================================================
# VALIDACION CRUZADA EXTERNA: LEAVE-ONE-OUT
# ============================================================================
# Como tenemos pocos sujetos, usamos leave-one-out cross-validation.
# En cada iteracion:
# 1) dejamos un sujeto afuera
# 2) en train elegimos alpha y lambda
# 3) ajustamos el modelo
# 4) predecimos el sujeto de test
n <- nrow(data_model)

predictions <- rep(NA_real_, n)
best_alpha_each_fold <- rep(NA_real_, n)
best_lambda_each_fold <- rep(NA_real_, n)

for (i in seq_len(n)) {
  
  # --------------------------------------------------------------------------
  # SPLIT TRAIN / TEST
  # --------------------------------------------------------------------------
  test_idx <- i
  train_idx <- setdiff(seq_len(n), test_idx)
  
  x_train <- x_full[train_idx, , drop = FALSE]
  y_train <- y_full[train_idx]
  
  x_test <- x_full[test_idx, , drop = FALSE]
  
  # --------------------------------------------------------------------------
  # TUNING INTERNO DE ALPHA
  # --------------------------------------------------------------------------
  # Para cada alpha, usamos cv.glmnet sobre train
  cv_errors <- rep(NA_real_, length(alpha_grid))
  cv_objects <- vector("list", length(alpha_grid))
  
  for (a in seq_along(alpha_grid)) {
    alpha_val <- alpha_grid[a]
    
    cv_fit <- cv.glmnet(
      x = x_train,
      y = y_train,
      alpha = alpha_val,
      nfolds = length(train_idx),
      standardize = TRUE,
      family = "gaussian"
    )
    
    cv_objects[[a]] <- cv_fit
    cv_errors[a] <- min(cv_fit$cvm)
  }
  
  # Elegimos el alpha con menor error de CV
  best_alpha_index <- which.min(cv_errors)
  best_alpha <- alpha_grid[best_alpha_index]
  best_cv <- cv_objects[[best_alpha_index]]
  best_lambda <- best_cv$lambda.min
  
  best_alpha_each_fold[i] <- best_alpha
  best_lambda_each_fold[i] <- best_lambda
  
  # --------------------------------------------------------------------------
  # AJUSTE FINAL EN TRAIN
  # --------------------------------------------------------------------------
  final_fit <- glmnet(
    x = x_train,
    y = y_train,
    alpha = best_alpha,
    lambda = best_lambda,
    standardize = TRUE,
    family = "gaussian"
  )
  
  # --------------------------------------------------------------------------
  # PREDICCION DEL SUJETO DE TEST
  # --------------------------------------------------------------------------
  predictions[i] <- as.numeric(predict(final_fit, newx = x_test, s = best_lambda))
}

# ============================================================================
# METRICAS DE PERFORMANCE EXTERNA
# ============================================================================
rmse_cv <- rmse_fun(y_full, predictions)
mae_cv  <- mae_fun(y_full, predictions)
r2_cv   <- r2_fun(y_full, predictions)
cor_cv  <- cor(y_full, predictions)

# ============================================================================
# AJUSTE FINAL SOBRE TODO EL DATASET
# ============================================================================
# Una vez evaluada la performance externa, ajustamos un modelo final sobre
# todos los sujetos para inspeccionar coeficientes
cv_errors_full <- rep(NA_real_, length(alpha_grid))
cv_objects_full <- vector("list", length(alpha_grid))

for (a in seq_along(alpha_grid)) {
  alpha_val <- alpha_grid[a]
  
  cv_fit_full <- cv.glmnet(
    x = x_full,
    y = y_full,
    alpha = alpha_val,
    nfolds = n,
    standardize = TRUE,
    family = "gaussian"
  )
  
  cv_objects_full[[a]] <- cv_fit_full
  cv_errors_full[a] <- min(cv_fit_full$cvm)
}

best_alpha_index_full <- which.min(cv_errors_full)
best_alpha_full <- alpha_grid[best_alpha_index_full]
best_cv_full <- cv_objects_full[[best_alpha_index_full]]
best_lambda_full <- best_cv_full$lambda.min

final_model_full <- glmnet(
  x = x_full,
  y = y_full,
  alpha = best_alpha_full,
  lambda = best_lambda_full,
  standardize = TRUE,
  family = "gaussian"
)

# ============================================================================
# COEFICIENTES DEL MODELO FINAL
# ============================================================================
coef_mat <- as.matrix(coef(final_model_full))
coef_df <- data.frame(
  term = rownames(coef_mat),
  coefficient = as.numeric(coef_mat[, 1]),
  row.names = NULL
)

coef_nonzero <- coef_df %>%
  filter(coefficient != 0)

# ============================================================================
# TABLA DE PREDICCIONES
# ============================================================================
pred_df <- data.frame(
  subject = data_model$subject,
  observed = y_full,
  predicted = predictions,
  residual = y_full - predictions,
  alpha_selected = best_alpha_each_fold,
  lambda_selected = best_lambda_each_fold
)

# ============================================================================
# IMPRESION DE RESULTADOS EN CONSOLA
# ============================================================================
cat("\n============================================================\n")
cat("RESULTADOS ELASTIC NET - CONDICION 100 + GRUPO\n")
cat("============================================================\n")
cat("Variable target:", target_var, "\n")
cat("Numero de sujetos:", n, "\n")
cat("Predictores usados:\n")
print(colnames(x_full))

cat("\nPerformance externa (LOOCV):\n")
cat("RMSE:", round(rmse_cv, 4), "\n")
cat("MAE :", round(mae_cv, 4), "\n")
cat("R2  :", round(r2_cv, 4), "\n")
cat("Cor :", round(cor_cv, 4), "\n")

cat("\nMejor alpha en el ajuste final:", best_alpha_full, "\n")
cat("Mejor lambda en el ajuste final:", best_lambda_full, "\n")

cat("\nCoeficientes no nulos del modelo final:\n")
print(coef_nonzero)

# ============================================================================
# GRAFICO 1: OBSERVADO VS PREDICHO
# ============================================================================
plot_obs_pred <- ggplot(pred_df, aes(x = observed, y = predicted)) +
  geom_point(size = 3) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = paste0("Elastic Net (condition 100 + Grupo) - ", target_var, ": observado vs predicho"),
    x = "Valor observado",
    y = "Valor predicho"
  ) +
  theme_bw()

ggsave(
  filename = file.path(plots_dir, paste0("Observed_vs_Predicted_condition_", selected_condition, "_", target_var, ".png")),
  plot = plot_obs_pred,
  width = 7,
  height = 6,
  dpi = 300
)

# ============================================================================
# GRAFICO 2: RESIDUOS VS PREDICHO
# ============================================================================
plot_resid <- ggplot(pred_df, aes(x = predicted, y = residual)) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = paste0("Elastic Net (condition 100 + Grupo) - ", target_var, ": residuos vs predicho"),
    x = "Valor predicho",
    y = "Residuo"
  ) +
  theme_bw()

ggsave(
  filename = file.path(plots_dir, paste0("Residuals_vs_Predicted_condition_", selected_condition, "_", target_var, ".png")),
  plot = plot_resid,
  width = 7,
  height = 6,
  dpi = 300
)

# ============================================================================
# GRAFICO 3: ERROR DE CV SEGUN ALPHA
# ============================================================================
alpha_df <- data.frame(
  alpha = alpha_grid,
  cv_error = cv_errors_full
)

plot_alpha <- ggplot(alpha_df, aes(x = alpha, y = cv_error)) +
  geom_point(size = 3) +
  geom_line() +
  labs(
    title = paste0("Error de validacion segun alpha - condition ", selected_condition, " - ", target_var),
    x = "alpha",
    y = "Error CV minimo"
  ) +
  theme_bw()

ggsave(
  filename = file.path(plots_dir, paste0("CV_Error_by_Alpha_condition_", selected_condition, "_", target_var, ".png")),
  plot = plot_alpha,
  width = 7,
  height = 6,
  dpi = 300
)

# ============================================================================
# GRAFICO 4: COEFICIENTES NO NULOS
# ============================================================================
coef_plot_df <- coef_nonzero %>%
  filter(term != "(Intercept)") %>%
  mutate(abs_coef = abs(coefficient)) %>%
  arrange(abs_coef)

if (nrow(coef_plot_df) > 0) {
  plot_coef <- ggplot(coef_plot_df, aes(x = reorder(term, abs_coef), y = coefficient)) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste0("Coeficientes no nulos - condition ", selected_condition, " - ", target_var),
      x = "Predictor",
      y = "Coeficiente"
    ) +
    theme_bw()
  
  ggsave(
    filename = file.path(plots_dir, paste0("Nonzero_Coefficients_condition_", selected_condition, "_", target_var, ".png")),
    plot = plot_coef,
    width = 8,
    height = 6,
    dpi = 300
  )
}

# ============================================================================
# EXPORT A EXCEL
# ============================================================================
wb <- createWorkbook()

addWorksheet(wb, "data_model")
writeData(wb, "data_model", data_model)

addWorksheet(wb, "predictions")
writeData(wb, "predictions", pred_df)

addWorksheet(wb, "coefficients_all")
writeData(wb, "coefficients_all", coef_df)

addWorksheet(wb, "coefficients_nonzero")
writeData(wb, "coefficients_nonzero", coef_nonzero)

addWorksheet(wb, "alpha_tuning")
writeData(wb, "alpha_tuning", alpha_df)

saveWorkbook(wb, output_excel, overwrite = TRUE)

# ============================================================================
# GUARDADO DEL RESUMEN EN TXT
# ============================================================================
sink(output_txt)

cat("ELASTIC NET SOBRE SCORES DE PCA - SOLO CONDICION 100 + GRUPO\n")
cat("============================================================\n\n")

cat("Variable target:", target_var, "\n")
cat("Condicion usada:", selected_condition, "\n")
cat("Sujetos excluidos:", paste(subjects_to_exclude, collapse = ", "), "\n")
cat("Numero de sujetos usados:", n, "\n\n")

cat("Predictores utilizados:\n")
print(colnames(x_full))
cat("\n")

cat("Performance externa (LOOCV):\n")
cat("RMSE:", round(rmse_cv, 6), "\n")
cat("MAE :", round(mae_cv, 6), "\n")
cat("R2  :", round(r2_cv, 6), "\n")
cat("Cor :", round(cor_cv, 6), "\n\n")

cat("Mejor alpha en el ajuste final:", best_alpha_full, "\n")
cat("Mejor lambda en el ajuste final:", best_lambda_full, "\n\n")

cat("Coeficientes completos del modelo final:\n")
print(coef_df)
cat("\n")

cat("Coeficientes no nulos del modelo final:\n")
print(coef_nonzero)
cat("\n")

sink()

# ============================================================================
# MENSAJES FINALES
# ============================================================================
cat("\nAnalisis terminado.\n")
cat("Archivo Excel:", output_excel, "\n")
cat("Archivo TXT  :", output_txt, "\n")
cat("Graficos en  :", plots_dir, "\n")