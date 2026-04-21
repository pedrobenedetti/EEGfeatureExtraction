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
# ggplot2: graficos
# openxlsx: exportar resultados a Excel
suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(openxlsx)
})

# ============================================================================
# CONFIGURACION GENERAL
# ============================================================================
# Archivos de entrada
pca_file <- "EEG_PCA_results.xlsx"
conduct_file <- "Protocolo2023_conducta.xlsx"

# Hojas a leer
pca_sheet <- "scores"
conduct_sheet <- "Hoja1"

# Variable conductual a correlacionar
target_var <- "REY"

# Sujetos a excluir
subjects_to_exclude <- c("02_test_2023", "15_test_2023")

# Componentes y condiciones a analizar
pcs_to_analyze <- c("PC1", "PC2", "PC3")
conditions_to_analyze <- c(40, 60, 100)

# Carpeta de salida para graficos
plots_dir <- "Correlations_PCs_REY"
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir)
}

# Archivos de salida
output_excel <- "PC_REY_correlations.xlsx"
output_txt <- "PC_REY_correlations_summary.txt"

# ============================================================================
# LECTURA DE DATOS
# ============================================================================
# Leemos scores del PCA
pca_scores <- read_excel(pca_file, sheet = pca_sheet)

# Leemos tabla conductual
conduct <- read_excel(conduct_file, sheet = conduct_sheet)

# ============================================================================
# MERGE ENTRE PCA Y CONDUCTA
# ============================================================================
# Unimos por subject y nos quedamos con:
# subject, Grupo, condition, PC1, PC2, PC3 y REY
data_merged <- inner_join(pca_scores, conduct, by = "subject") %>%
  filter(!subject %in% subjects_to_exclude) %>%
  select(subject, Grupo, condition, all_of(pcs_to_analyze), all_of(target_var))

# Convertimos variables categoricas
data_merged <- data_merged %>%
  mutate(
    subject = as.factor(subject),
    Grupo = factor(Grupo, levels = c("Habituación", "Novedad")),
    condition = as.factor(condition)
  )

# ============================================================================
# CHEQUEO BASICO
# ============================================================================
cat("Primeras filas del dataset:\n")
print(head(data_merged))

cat("\nCantidad total de filas:\n")
print(nrow(data_merged))

cat("\nCantidad de sujetos unicos:\n")
print(length(unique(data_merged$subject)))

# ============================================================================
# OBJETOS PARA GUARDAR RESULTADOS
# ============================================================================
results_list <- list()

# ============================================================================
# INICIO DEL ARCHIVO DE TEXTO
# ============================================================================
cat("CORRELACIONES ENTRE PCs Y REY\n",
    file = output_txt)
cat("============================================================\n\n",
    file = output_txt, append = TRUE)

cat(paste0("Variable conductual: ", target_var, "\n"),
    file = output_txt, append = TRUE)
cat(paste0("Sujetos excluidos: ", paste(subjects_to_exclude, collapse = ", "), "\n\n"),
    file = output_txt, append = TRUE)

# ============================================================================
# BUCLE PRINCIPAL: 9 ANALISIS
# ============================================================================
# Recorremos condiciones y PCs
for (cond in conditions_to_analyze) {
  for (pc in pcs_to_analyze) {
    
    # ------------------------------------------------------------------------
    # FILTRO DE DATOS PARA ESA CONDICION
    # ------------------------------------------------------------------------
    data_sub <- data_merged %>%
      filter(condition == as.character(cond)) %>%
      select(subject, Grupo, condition, all_of(pc), all_of(target_var)) %>%
      drop_na()
    
    # ------------------------------------------------------------------------
    # TEST DE CORRELACION
    # ------------------------------------------------------------------------
    # Usamos Pearson
    cor_test <- cor.test(data_sub[[pc]], data_sub[[target_var]], method = "pearson")
    
    # Guardamos resultados en una tabla
    result_row <- data.frame(
      condition = cond,
      PC = pc,
      n = nrow(data_sub),
      r = unname(cor_test$estimate),
      p_value = cor_test$p.value,
      CI_low = cor_test$conf.int[1],
      CI_high = cor_test$conf.int[2],
      stringsAsFactors = FALSE
    )
    
    results_list[[paste0(pc, "_", cond)]] <- result_row
    
    # ------------------------------------------------------------------------
    # IMPRESION EN CONSOLA
    # ------------------------------------------------------------------------
    cat("\n============================================================\n")
    cat(paste0("CORRELACION: ", pc, " vs ", target_var, " | condition ", cond, "\n"))
    cat("============================================================\n")
    print(cor_test)
    
    # ------------------------------------------------------------------------
    # ESCRITURA EN TXT
    # ------------------------------------------------------------------------
    cat(paste0("CORRELACION: ", pc, " vs ", target_var, " | condition ", cond, "\n"),
        file = output_txt, append = TRUE)
    cat("------------------------------------------------------------\n",
        file = output_txt, append = TRUE)
    capture.output(print(cor_test), file = output_txt, append = TRUE)
    cat("\n", file = output_txt, append = TRUE)
    
    # ------------------------------------------------------------------------
    # GRAFICO
    # ------------------------------------------------------------------------
    # Scatter con:
    # - puntos de todos los sujetos
    # - color por grupo
    # - recta global de regresion
    # - etiqueta con r y p
    label_text <- paste0(
      "r = ", round(unname(cor_test$estimate), 3),
      "\np = ", signif(cor_test$p.value, 3),
      "\nn = ", nrow(data_sub)
    )
    
    plot_cor <- ggplot(data_sub, aes(x = .data[[pc]], y = .data[[target_var]], color = Grupo)) +
      geom_point(size = 3, alpha = 0.9) +
      geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.8) +
      scale_color_manual(values = c("Habituación" = "#008080", "Novedad" = "#dc143c")) +
      annotate(
        "text",
        x = Inf, y = Inf,
        label = label_text,
        hjust = 1.1, vjust = 1.2,
        size = 4
      ) +
      labs(
        title = paste0(pc, " vs ", target_var, " | condition ", cond),
        x = pc,
        y = target_var,
        color = "Grupo"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5)
      )
    
    ggsave(
      filename = file.path(plots_dir, paste0(pc, "_condition_", cond, "_vs_", target_var, ".png")),
      plot = plot_cor,
      width = 7,
      height = 6,
      dpi = 300
    )
  }
}

# ============================================================================
# TABLA FINAL DE RESULTADOS
# ============================================================================
results_df <- bind_rows(results_list)

# Corregimos por comparaciones multiples sobre los 9 tests
results_df <- results_df %>%
  mutate(
    p_holm = p.adjust(p_value, method = "holm"),
    significant_raw = p_value < 0.05,
    significant_holm = p_holm < 0.05
  ) %>%
  arrange(condition, PC)

# ============================================================================
# IMPRESION FINAL EN CONSOLA
# ============================================================================
cat("\n============================================================\n")
cat("TABLA FINAL DE CORRELACIONES\n")
cat("============================================================\n")
print(results_df)

# ============================================================================
# GUARDADO A EXCEL
# ============================================================================
wb <- createWorkbook()

addWorksheet(wb, "correlations")
writeData(wb, "correlations", results_df)

addWorksheet(wb, "data_used")
writeData(wb, "data_used", data_merged)

saveWorkbook(wb, output_excel, overwrite = TRUE)

# ============================================================================
# AGREGAMOS TABLA FINAL AL TXT
# ============================================================================
cat("TABLA FINAL DE CORRELACIONES\n",
    file = output_txt, append = TRUE)
cat("------------------------------------------------------------\n",
    file = output_txt, append = TRUE)
capture.output(print(results_df), file = output_txt, append = TRUE)
cat("\n", file = output_txt, append = TRUE)

# ============================================================================
# MENSAJES FINALES
# ============================================================================
cat("\nAnalisis terminado.\n")
cat("Archivo Excel:", output_excel, "\n")
cat("Archivo TXT  :", output_txt, "\n")
# cat("Graficos en  :", plots_dir, "\n")