library(tidyverse)

expit = function(x) {
  exp(x) / (1 + exp(x))
}


data_folder = "simple_simulated_data/out/"

data_true = read.csv(file.path(data_folder, "data_true.csv"), row.names=1) %>% as_tibble() %>% 
  mutate(mutation_ids=paste0("M", mutation_ids))

mutation_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% pull(mutation_ids)
cell_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% select(-mutation_ids) %>% 
  colnames

losses_grads = read.csv(file.path(data_folder, "losses_kernel.csv")) %>% mutate(type="w_kernel") %>% 
  bind_rows(read.csv(file.path(data_folder, "losses_Nkernel.csv")) %>% mutate(type="wout_kernel")) %>% 
  
  left_join(
    read.csv(file.path(data_folder, "grads_kernel.csv")) %>% mutate(type="w_kernel") %>% 
      bind_rows(read.csv(file.path(data_folder, "grads_Nkernel.csv")) %>% mutate(type="wout_kernel"))
  )

losses_grads %>% 
  ggplot() +
  geom_line(aes(x=X, y=losses, color=type)) + facet_wrap(~type, scales="free")

losses_grads %>% 
  ggplot() +
  geom_line(aes(x=X, y=grads, color=type)) + facet_wrap(~type, scales="free")

load_matrix = function(file_name, row_names, cell_ids) {
  tmp = read.csv(file_name, row.names=1, check.names=F)
  rownames(tmp) = row_names
  colnames(tmp) = cell_ids
  return(tmp)
}

w_kernel = list(
  mu_init = load_matrix(file.path(data_folder, "mu_init_kernel.csv"), mutation_ids, cell_ids),
  gamma_hat = load_matrix(file.path(data_folder, "gamma_hat_kernel.csv"), mutation_ids, cell_ids),
  mu = load_matrix(file.path(data_folder, "mu_kernel.csv"), mutation_ids, cell_ids)
)
w_kernel$theta = expit(w_kernel$gamma_hat)

wout_kernel = list(
  mu_init = load_matrix(file.path(data_folder, "mu_init_Nkernel.csv"), mutation_ids, cell_ids),
  gamma_hat = load_matrix(file.path(data_folder, "gamma_hat_Nkernel.csv"), mutation_ids, cell_ids),
  mu = load_matrix(file.path(data_folder, "mu_Nkernel.csv"), mutation_ids, cell_ids)
)
wout_kernel$theta = expit(wout_kernel$gamma_hat)

final_vafs = w_kernel$theta %>% tibble::rownames_to_column("mutation_ids") %>% 
  pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
  mutate(type="w_kernel") %>% 
  
  bind_rows(
    wout_kernel$theta %>% tibble::rownames_to_column("mutation_ids") %>% 
      pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
      mutate(type="wout_kernel") 
  )



VAF = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% 
  column_to_rownames("mutation_ids")

VAF_inferred_wkernel = final_vafs %>% 
  filter(type=="w_kernel") %>% 
  select(mutation_ids, cell_ids, VAF_inf) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF_inf") %>% 
  column_to_rownames("mutation_ids")

VAF_inferred_woutkernel = final_vafs %>% 
  filter(type=="wout_kernel") %>% 
  select(mutation_ids, cell_ids, VAF_inf) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF_inf") %>% 
  column_to_rownames("mutation_ids")


library(dendextend)
library(ComplexHeatmap)
distances = read.csv(file.path(data_folder, "distances.csv"), row.names=1)
hclust_res = hclust(as.dist(distances))
row_dend = as.dendrogram(hclust_res)
assignments = data_true %>% select(cell_ids, clone_ids) %>% unique() %>% column_to_rownames("cell_ids")

pl_vafs = list()
pl_vafs["observed"] = Heatmap(VAF[, hclust_res$labels], name="VAF",
                              cluster_columns=row_dend, 
                              cluster_rows=T,
                              top_annotation=HeatmapAnnotation(clone_id=assignments[hclust_res$labels, "clone_ids"],
                                                               col=list(clone_id=c("#5F9EA0","#FF4500") %>% setNames(c("1","2"))),
                                                               show_annotation_name=FALSE, show_legend=FALSE),
                              col=circlize::colorRamp2(c(0,1), c("white","steelblue4")),
                              show_column_names=FALSE,
                              show_row_names=FALSE, show_row_dend=FALSE,
                              column_dend_height=unit(2, "cm"))

# pl_vafs["inferred_wkernel"] = Heatmap(VAF_inferred_wkernel[, hclust_res$labels], name="VAF",
#                               cluster_columns=row_dend, 
#                               cluster_rows=T,
#                               top_annotation=HeatmapAnnotation(clone_id=assignments[hclust_res$labels, "clone_ids"],
#                                                                col=list(clone_id=c("#5F9EA0","#FF4500") %>% setNames(c("1","2"))),
#                                                                show_annotation_name=FALSE, show_legend=FALSE),
#                               col=circlize::colorRamp2(c(0,1), c("white","steelblue4")),
#                               show_column_names=FALSE,
#                               show_row_names=FALSE, show_row_dend=FALSE,
#                               column_dend_height=unit(2, "cm"))
# 
# pl_vafs["inferred_woutkernel"] = Heatmap(VAF_inferred_woutkernel[, hclust_res$labels], name="VAF",
#                                       cluster_columns=row_dend, 
#                                       cluster_rows=T,
#                                       top_annotation=HeatmapAnnotation(clone_id=assignments[hclust_res$labels, "clone_ids"],
#                                                                        col=list(clone_id=c("#5F9EA0","#FF4500") %>% setNames(c("1","2"))),
#                                                                        show_annotation_name=FALSE, show_legend=FALSE),
#                                       col=circlize::colorRamp2(c(0,1), c("white","steelblue4")),
#                                       show_column_names=FALSE,
#                                       show_row_names=FALSE, show_row_dend=FALSE,
#                                       column_dend_height=unit(2, "cm"))






input_d = final_vafs %>% 
  left_join(data_true) %>% 
  select(mutation_ids, cell_ids, VAF_inf, VAF, type) %>% 
  reshape2::melt(id=c("mutation_ids", "cell_ids", "type")) %>% as_tibble() %>% 
  mutate(variable=factor(variable, levels=c("VAF","VAF_inf")))

input_d %>% 
  ggplot() +
  geom_line(aes(x=variable, y=value, group=interaction(mutation_ids, cell_ids)))




