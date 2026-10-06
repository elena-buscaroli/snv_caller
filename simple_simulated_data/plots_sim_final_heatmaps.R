library(tidyverse)

expit = function(x) {
  exp(x) / (1 + exp(x))
}

load_matrix = function(file_name, row_names, cell_ids) {
  tmp = read.csv(file_name, row.names=1, check.names=F)
  rownames(tmp) = row_names
  colnames(tmp) = cell_ids
  return(tmp)
}

data_folder = "simple_simulated_data/out/"

data_true = read.csv(file.path(data_folder, "data_true.csv"), row.names=1) %>% as_tibble() %>% 
  mutate(mutation_ids=paste0("M", mutation_ids)) %>% 
  mutate(presence_dropout=ifelse(presence==1, "Present", "Absent"),
         presence=ifelse(AD>0, "Present", "Absent"))

mutation_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% pull(mutation_ids)
cell_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% select(-mutation_ids) %>% 
  colnames

VAF_inf = load_matrix(file.path(data_folder, "gamma_hat_kernel.csv"), mutation_ids, cell_ids)
VAF = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% 
  column_to_rownames("mutation_ids")

write.csv(VAF, "simple_simulated_data/out/VAF_true.csv")


library(dendextend)
library(ComplexHeatmap)
distances = read.csv(file.path(data_folder, "distances.csv"), row.names=1)
hclust_res = hclust(as.dist(distances))
row_dend = as.dendrogram(hclust_res)
assignments = data_true %>% select(cell_ids, clone_ids) %>% unique() %>% column_to_rownames("cell_ids")

pl_heatmap_VAF = Heatmap(as.matrix(VAF)[, hclust_res$labels], name="VAF",
        cluster_columns=row_dend, 
        cluster_rows=T,
        top_annotation=HeatmapAnnotation(clone_id=assignments[hclust_res$labels, "clone_ids"],
                                         col=list(clone_id=c("#5F9EA0","#FF4500") %>% setNames(c("1","2"))),
                                         show_annotation_name=FALSE, show_legend=FALSE),
        col=circlize::colorRamp2(c(0,1), c("white","steelblue4")),
        show_column_names=FALSE,
        show_row_names=FALSE, show_row_dend=TRUE,
        column_dend_height=unit(2, "cm"))


kernel = read.csv("simple_simulated_data/out/kernel.csv", row.names=1)
colnames(kernel) = rownames(kernel) = cell_ids
pl_heatmap_kernel = Heatmap(as.matrix(kernel)[hclust_res$labels, hclust_res$labels], name="Kernel",
                         cluster_columns=row_dend, 
                         cluster_rows=row_dend,
                         # cluster_rows=T,
                         top_annotation=HeatmapAnnotation(clone_id=assignments[hclust_res$labels, "clone_ids"],
                                                          col=list(clone_id=c("#5F9EA0","#FF4500") %>% setNames(c("1","2"))),
                                                          show_annotation_name=FALSE, show_legend=FALSE),
                         col=circlize::colorRamp2(c(0.75,1), c("white","steelblue4")),
                         show_column_names=FALSE,
                         show_row_names=FALSE,
                         column_dend_height=unit(2, "cm"),
                         column_dend_height=unit(2, "cm"))




