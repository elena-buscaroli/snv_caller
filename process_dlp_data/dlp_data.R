# remotes::install_github("caravagnalab/ProCESS", ref="on-the-fly_mutations")
library(ProCESS)
library(ape)
# library(igraph)
library(tidyverse)

AD_true = readRDS("process_dlp_data/somatic_nv.rds") %>% 
  pivot_wider(id_cols="mutationID", names_from="sample", values_from="NV", values_fill=0) %>% 
  column_to_rownames(var="mutationID")
DP_true = readRDS("process_dlp_data/somatic_dp.rds") %>% 
  pivot_wider(id_cols="mutationID", names_from="sample", values_from="DP", values_fill=0) %>% 
  column_to_rownames(var="mutationID")

write.csv(AD_true, "process_dlp_data/AD_true.csv")
write.csv(DP_true, "process_dlp_data/DP_true.csv")

heatmap(as.matrix(AD_true[1:100,]))

sample_forest = load_sample_forest("process_dlp_data/sample_forest_1.sff")
phylo_forest = load_phylogenetic_forest("process_dlp_data/phylo_forest_1.sff")

sample_forest %>% plot_forest() + theme(legend.position="none")

nodes = sample_forest$get_nodes()
mutations = phylo_forest$get_sampled_cell_mutations()

saveRDS(nodes, "process_dlp_data/nodes.rds")
saveRDS(mutations, "process_dlp_data/mutations.rds")




library(ape)
library(dplyr)

nodes = nodes %>% 
  mutate(ancestor=replace(ancestor, is.na(ancestor), "root")) %>% 
  mutate(cell_id=factor(cell_id),
         ancestor=factor(ancestor))

root = setdiff(nodes$ancestor, nodes$cell_id)

edges = nodes %>%
  filter(!is.na(ancestor)) %>%
  mutate(length=birth_time - nodes$birth_time[match(ancestor, nodes$cell_id)]) %>% 
  select(ancestor, cell_id, length)

tree = as.phylo.formula(x = ~ancestor/cell_id, data=nodes)
tree$edge.length <- edges$length

D = cophenetic.phylo(tree)
