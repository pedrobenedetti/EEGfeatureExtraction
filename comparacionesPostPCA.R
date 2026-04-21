# ============================================================================
# SCRIPT: Analisis mixto de PC1, PC2 y PC3 por Grupo y Condicion + graficos
#         excluyendo outliers por criterio 1.5*IQR
# ============================================================================
# Este script:
# 1) Lee los scores de PCA desde EEG_PCA_results.xlsx
# 2) Lee los datos conductuales desde Protocolo2023_conducta.xlsx
# 3) Hace merge por la columna subject
# 4) Excluye a los sujetos 02_test_2023 y 15_test_2023
# 5) Detecta outliers en PC1, PC2 y PC3 usando 1.5*IQR dentro de cada condicion
# 6) Imprime a que sujetos corresponden esos outliers
# 7) Excluye esos outliers del analisis
# 8) Ajusta 3 modelos lineales mixtos:
#       PC1 ~ Grupo * condition + (1 | subject)
#       PC2 ~ Grupo * condition + (1 | subject)
#       PC3 ~ Grupo * condition + (1 | subject)
# 9) Evalua efectos principales e interaccion
# 10) Si hay interaccion significativa, hace comparaciones post hoc
# 11) Guarda tablas y resumenes en archivos
# 12) Genera graficos para PC1, PC2 y PC3
# 13) Imprime en consola los resultados de los modelos y de las comparaciones
# ============================================================================

# ============================================================================
# CARGA DE LIBRERIAS
# ============================================================================
library(readxl)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(openxlsx)
library(broom.mixed)
library(ggplot2)


rm(list = ls())
cat("\014")
graphics.off()
setwd("/media/pedro/Expansion/Doctorado/protocol2023")

# ============================================================================
# CONFIGURACION DE ARCHIVOS
# ============================================================================
pca_file <- "EEG_PCA_results.xlsx"
conduct_file <- "Protocolo2023_conducta.xlsx"

output_excel <- "PC_mixed_models_results_no_outliers.xlsx"
output_txt <- "PC_mixed_models_summary_no_outliers.txt"

plots_dir <- "PC_plots_no_outliers"

# ============================================================================
# CREACION DE CARPETA PARA GRAFICOS
# ============================================================================
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir)
}

# ============================================================================
# LECTURA DE DATOS DE PCA
# ============================================================================
pca_scores <- read_excel(pca_file, sheet = "scores")

# ============================================================================
# LECTURA DE DATOS CONDUCTUALES
# ============================================================================
conduct <- read_excel(conduct_file, sheet = "Hoja1")

# ============================================================================
# INSPECCION BASICA DE COLUMNAS
# ============================================================================
cat("\nColumnas de PCA:\n")
print(colnames(pca_scores))

cat("\nColumnas de conducta:\n")
print(colnames(conduct))

# ============================================================================
# MERGE ENTRE LOS DATOS DE PCA Y CONDUCTA
# ============================================================================
data_merged <- inner_join(pca_scores, conduct, by = "subject")

# ============================================================================
# EXCLUSION DE SUJETOS
# ============================================================================
subjects_to_exclude <- c("02_test_2023", "15_test_2023")

data_merged <- data_merged %>%
  filter(!subject %in% subjects_to_exclude)

# ============================================================================
# PREPARACION DE VARIABLES
# ============================================================================
data_merged <- data_merged %>%
  mutate(
    subject = as.factor(subject),
    Grupo = as.factor(Grupo),
    condition = as.factor(condition)
  )

# ============================================================================
# CHEQUEO DEL DATASET FINAL ANTES DE OUTLIERS
# ============================================================================
cat("\nPrimeras filas del dataset mergeado:\n")
print(head(data_merged))

cat("\nEstructura del dataset mergeado:\n")
print(str(data_merged))

cat("\nSujetos excluidos manualmente:\n")
print(subjects_to_exclude)

cat("\nCantidad de sujetos restantes antes de quitar outliers:\n")
print(length(unique(data_merged$subject)))

# ============================================================================
# COMPONENTES A ANALIZAR
# ============================================================================
pcs_to_analyze <- c("PC1", "PC2", "PC3")

# ============================================================================
# DETECCION DE OUTLIERS POR 1.5*IQR
# ============================================================================
# Vamos a detectar outliers por cada PC dentro de cada condicion.
# Esto evita mezclar condiciones con distribuciones distintas.
outliers_list <- list()

for (pc in pcs_to_analyze) {
  
  outliers_pc <- data_merged %>%
    group_by(condition) %>%
    mutate(
      Q1 = quantile(.data[[pc]], 0.25, na.rm = TRUE),
      Q3 = quantile(.data[[pc]], 0.75, na.rm = TRUE),
      IQR_value = IQR(.data[[pc]], na.rm = TRUE),
      lower_bound = Q1 - 1.5 * IQR_value,
      upper_bound = Q3 + 1.5 * IQR_value,
      is_outlier = .data[[pc]] < lower_bound | .data[[pc]] > upper_bound
    ) %>%
    ungroup() %>%
    filter(is_outlier) %>%
    select(subject, Grupo, condition, all_of(pc), lower_bound, upper_bound)
  
  outliers_pc$PC <- pc
  outliers_list[[pc]] <- outliers_pc
}

# Unimos todos los outliers detectados
outliers_df <- bind_rows(outliers_list)

# ============================================================================
# IMPRESION DE OUTLIERS EN CONSOLA
# ============================================================================
cat("\n")
cat("============================================================\n")
cat("OUTLIERS DETECTADOS POR CRITERIO 1.5*IQR\n")
cat("============================================================\n")

if (nrow(outliers_df) == 0) {
  cat("\nNo se detectaron outliers.\n")
} else {
  print(outliers_df)
}

# ============================================================================
# GUARDADO DE OUTLIERS EN TXT
# ============================================================================
cat("ANALISIS MIXTO DE PC1, PC2 Y PC3 SIN OUTLIERS\n",
    file = output_txt)
cat("============================================================\n\n",
    file = output_txt, append = TRUE)

cat(paste0("Cantidad de filas antes de quitar outliers: ", nrow(data_merged), "\n"),
    file = output_txt, append = TRUE)
cat(paste0("Cantidad de sujetos unicos antes de quitar outliers: ", length(unique(data_merged$subject)), "\n"),
    file = output_txt, append = TRUE)
cat(paste0("Sujetos excluidos manualmente: ", paste(subjects_to_exclude, collapse = ", "), "\n\n"),
    file = output_txt, append = TRUE)

cat("OUTLIERS DETECTADOS POR CRITERIO 1.5*IQR\n",
    file = output_txt, append = TRUE)
cat("------------------------------------------------------------\n",
    file = output_txt, append = TRUE)

if (nrow(outliers_df) == 0) {
  cat("No se detectaron outliers.\n\n",
      file = output_txt, append = TRUE)
} else {
  capture.output(print(outliers_df), file = output_txt, append = TRUE)
  cat("\n\n", file = output_txt, append = TRUE)
}

# ============================================================================
# EXCLUSION DE OUTLIERS DEL ANALISIS
# ============================================================================
# Excluimos filas que fueron marcadas como outliers en cualquiera de las PCs.
# Ojo: esto excluye la observacion sujeto-condicion puntual, no al sujeto entero.
if (nrow(outliers_df) == 0) {
  data_clean <- data_merged
} else {
  outlier_keys <- outliers_df %>%
    distinct(subject, condition)
  
  data_clean <- anti_join(data_merged, outlier_keys, by = c("subject", "condition"))
}

# ============================================================================
# CHEQUEO DEL DATASET LIMPIO
# ============================================================================
cat("\nCantidad de filas despues de quitar outliers:\n")
print(nrow(data_clean))

cat("\nCantidad de sujetos unicos despues de quitar outliers:\n")
print(length(unique(data_clean$subject)))

cat(paste0("\nCantidad total de filas eliminadas por outliers: ", nrow(data_merged) - nrow(data_clean), "\n"))

cat(paste0("Cantidad de filas despues de quitar outliers: ", nrow(data_clean), "\n"),
    file = output_txt, append = TRUE)
cat(paste0("Cantidad de sujetos unicos despues de quitar outliers: ", length(unique(data_clean$subject)), "\n"),
    file = output_txt, append = TRUE)
cat(paste0("Cantidad total de filas eliminadas por outliers: ", nrow(data_merged) - nrow(data_clean), "\n\n"),
    file = output_txt, append = TRUE)

# ============================================================================
# OBJETOS PARA GUARDAR RESULTADOS
# ============================================================================
results_models <- list()
results_anova <- list()
results_fixed <- list()
results_emmeans <- list()

# ============================================================================
# BUCLE PRINCIPAL: UN MODELO POR CADA PC
# ============================================================================
for (pc in pcs_to_analyze) {
  
  # --------------------------------------------------------------------------
  # FORMULA DEL MODELO
  # --------------------------------------------------------------------------
  formula_pc <- as.formula(paste0(pc, " ~ Grupo * condition + (1 | subject)"))
  
  # --------------------------------------------------------------------------
  # AJUSTE DEL MODELO
  # --------------------------------------------------------------------------
  model <- lmer(formula_pc, data = data_clean)
  results_models[[pc]] <- model
  
  # --------------------------------------------------------------------------
  # TABLA ANOVA DEL MODELO
  # --------------------------------------------------------------------------
  anova_pc <- anova(model)
  results_anova[[pc]] <- anova_pc
  
  # --------------------------------------------------------------------------
  # COEFICIENTES FIJOS DEL MODELO
  # --------------------------------------------------------------------------
  fixed_pc <- broom.mixed::tidy(model, effects = "fixed")
  results_fixed[[pc]] <- fixed_pc
  
  # --------------------------------------------------------------------------
  # IMPRESION EN CONSOLA DEL MODELO
  # --------------------------------------------------------------------------
  cat("\n")
  cat("============================================================\n")
  cat(paste0("RESULTADOS PARA ", pc, " (SIN OUTLIERS)\n"))
  cat("============================================================\n")
  
  cat("\nFormula del modelo:\n")
  print(formula_pc)
  
  cat("\nResumen del modelo:\n")
  print(summary(model))
  
  cat("\nANOVA del modelo:\n")
  print(anova_pc)
  
  cat("\nCoeficientes fijos:\n")
  print(fixed_pc)
  
  # --------------------------------------------------------------------------
  # ESCRITURA DEL RESUMEN EN TXT
  # --------------------------------------------------------------------------
  cat(paste0("RESULTADOS PARA ", pc, " (SIN OUTLIERS)\n"),
      file = output_txt, append = TRUE)
  cat("------------------------------------------------------------\n",
      file = output_txt, append = TRUE)
  
  cat("Formula del modelo:\n",
      file = output_txt, append = TRUE)
  cat(paste0(pc, " ~ Grupo * condition + (1 | subject)\n\n"),
      file = output_txt, append = TRUE)
  
  cat("Resumen del modelo:\n",
      file = output_txt, append = TRUE)
  capture.output(print(summary(model)), file = output_txt, append = TRUE)
  
  cat("\nANOVA del modelo:\n",
      file = output_txt, append = TRUE)
  capture.output(print(anova_pc), file = output_txt, append = TRUE)
  
  cat("\nCoeficientes fijos:\n",
      file = output_txt, append = TRUE)
  capture.output(print(fixed_pc), file = output_txt, append = TRUE)
  
  cat("\n", file = output_txt, append = TRUE)
  
  # --------------------------------------------------------------------------
  # POST HOC SI LA INTERACCION ES SIGNIFICATIVA
  # --------------------------------------------------------------------------
  if ("Grupo:condition" %in% rownames(anova_pc)) {
    
    p_interaction <- anova_pc["Grupo:condition", "Pr(>F)"]
    
    if (!is.na(p_interaction) && p_interaction < 0.05) {
      
      emm_pc <- emmeans(model, ~ Grupo | condition)
      contrast_pc <- contrast(emm_pc, method = "pairwise", adjust = "bonferroni")
      
      results_emmeans[[pc]] <- as.data.frame(contrast_pc)
      
      cat("\nPost hoc: comparaciones entre grupos dentro de cada condicion\n")
      print(contrast_pc)
      
      cat("\nMedias marginales estimadas\n")
      print(emm_pc)
      
      cat("\n")
      
      cat(paste0("Post hoc para ", pc, " (interaccion significativa)\n"),
          file = output_txt, append = TRUE)
      cat("Comparaciones entre grupos dentro de cada condicion:\n",
          file = output_txt, append = TRUE)
      capture.output(print(contrast_pc), file = output_txt, append = TRUE)
      
      cat("\nMedias marginales estimadas:\n",
          file = output_txt, append = TRUE)
      capture.output(print(emm_pc), file = output_txt, append = TRUE)
      
      cat("\n\n", file = output_txt, append = TRUE)
      
    } else {
      
      cat("\nNo hubo interaccion significativa Grupo:condition, no se realiza post hoc.\n\n")
      
      cat(paste0("No se realizo post hoc para ", pc,
                 " porque la interaccion Grupo:condition no fue significativa.\n\n"),
          file = output_txt, append = TRUE)
    }
  }
  
  # --------------------------------------------------------------------------
  # GRAFICO 1: BOXPLOT + JITTER POR GRUPO, FACETADO POR CONDICION
  # --------------------------------------------------------------------------
  plot_violin <- ggplot(data_clean, aes(x = Grupo, y = .data[[pc]], fill = Grupo, color = Grupo)) +
    geom_violin(trim = FALSE, alpha = 0.35) +
    geom_boxplot(width = 0.18, alpha = 0.5, outlier.shape = NA, color = "black") +
    geom_jitter(width = 0.12, size = 2, alpha = 0.8) +
    facet_wrap(~ condition) +
    scale_fill_manual(values = c("Habituación" = "#008080", "Novedad" = "#dc143c")) +
    scale_color_manual(values = c("Habituación" = "#008080", "Novedad" = "#dc143c")) +
    labs(
      title = paste0(pc, ": distribucion por Grupo y condition (sin outliers)"),
      x = "Grupo",
      y = pc
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "none"
    )
  
  ggsave(
    filename = file.path(plots_dir, paste0(pc, "_violin_by_group_condition_no_outliers.png")),
    plot = plot_violin,
    width = 10,
    height = 6,
    dpi = 300
  )
  
  # --------------------------------------------------------------------------
  # CALCULO DE RESUMENES PARA GRAFICO DE MEDIAS
  # --------------------------------------------------------------------------
  summary_pc <- data_clean %>%
    group_by(Grupo, condition) %>%
    summarise(
      mean_value = mean(.data[[pc]], na.rm = TRUE),
      sd_value = sd(.data[[pc]], na.rm = TRUE),
      n = n(),
      se_value = sd_value / sqrt(n),
      .groups = "drop"
    )
  
  # --------------------------------------------------------------------------
  # GRAFICO 2: MEDIAS POR GRUPO DENTRO DE CADA CONDICION
  # --------------------------------------------------------------------------
  plot_means <- ggplot(summary_pc, aes(x = condition, y = mean_value, color = Grupo, group = Grupo)) +
    geom_point(size = 3) +
    geom_line(linewidth = 1) +
    geom_errorbar(aes(ymin = mean_value - se_value, ymax = mean_value + se_value),
                  width = 0.1) +
    scale_color_manual(values = c("Habituación" = "#008080", "Novedad" = "#dc143c")) +
    labs(
      title = paste0(pc, ": medias por Grupo y condition (sin outliers)"),
      x = "condition",
      y = paste0(pc, " (media ± EE)")
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5)
    )
  
  ggsave(
    filename = file.path(plots_dir, paste0(pc, "_means_by_group_condition_no_outliers.png")),
    plot = plot_means,
    width = 8,
    height = 6,
    dpi = 300
  )
}

# ============================================================================
# EXPORTACION A EXCEL
# ============================================================================
wb <- createWorkbook()

addWorksheet(wb, "data_clean")
writeData(wb, "data_clean", data_clean)

addWorksheet(wb, "outliers_detected")
writeData(wb, "outliers_detected", outliers_df)

for (pc in pcs_to_analyze) {
  sheet_name <- paste0(pc, "_anova")
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, as.data.frame(results_anova[[pc]]), rowNames = TRUE)
}

for (pc in pcs_to_analyze) {
  sheet_name <- paste0(pc, "_fixed")
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, results_fixed[[pc]])
}

for (pc in names(results_emmeans)) {
  sheet_name <- paste0(pc, "_posthoc")
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, results_emmeans[[pc]])
}

saveWorkbook(wb, output_excel, overwrite = TRUE)

# ============================================================================
# RESUMEN GENERAL FINAL EN CONSOLA
# ============================================================================
cat("\n")
cat("============================================================\n")
cat("RESUMEN GENERAL DE EFECTOS (SIN OUTLIERS)\n")
cat("============================================================\n")

for (pc in pcs_to_analyze) {
  cat("\n", pc, "\n", sep = "")
  print(results_anova[[pc]])
}

# ============================================================================
# MENSAJES FINALES EN CONSOLA
# ============================================================================
cat("\nAnalisis completado.\n")
cat(paste0("Resumen guardado en: ", output_txt, "\n"))
cat(paste0("Tablas guardadas en: ", output_excel, "\n"))
cat(paste0("Graficos guardados en la carpeta: ", plots_dir, "\n"))