# remotes::install_github("caravagnalab/ProCESS", ref="on-the-fly_mutations")
library(ProCESS)
library(ape)
library(tidyverse)

AD_true = readRDS("process_dlp_data/high_cov_300x/somatic_nv_high_cov.rds") %>% 
  pivot_wider(id_cols="mutationID", names_from="sample", values_from="NV", values_fill=0) %>% 
  column_to_rownames(var="mutationID")
DP_true = readRDS("process_dlp_data/high_cov_1500x/somatic_dp_high_cov.rds") %>% 
  pivot_wider(id_cols="mutationID", names_from="sample", values_from="DP", values_fill=0) %>% 
  column_to_rownames(var="mutationID")

# write.csv(AD_true, "process_dlp_data/AD_true.csv")
# write.csv(DP_true, "process_dlp_data/DP_true.csv")

heatmap(as.matrix(AD_true[1:100,]))

# sample_forest = load_sample_forest("process_dlp_data/sample_forest_1.sff")
# phylo_forest = load_phylogenetic_forest("process_dlp_data/phylo_forest_1.sff")
# sample_forest %>% plot_forest() + theme(legend.position="none")
# nodes = sample_forest$get_nodes()
# mutations = phylo_forest$get_sampled_cell_mutations()
# saveRDS(nodes, "process_dlp_data/nodes.rds")
# saveRDS(mutations, "process_dlp_data/mutations.rds")

nodes_all = readRDS("process_dlp_data/nodes.rds") %>% as_tibble() %>% 
  mutate(ancestor=replace(ancestor, is.na(ancestor), "root"))
mutations = readRDS("process_dlp_data/mutations.rds") %>% as_tibble() %>% 
  mutate(mutationID=paste(chr, from, ref, alt, sep=":")) %>% 
  select(cell_id, mutationID, type, cause, class, allele)

nodes = nodes_all %>%
  mutate(ancestor=as.character(ancestor),
         cell_id=as.character(cell_id))

edges = nodes %>%
  left_join(nodes %>% select(ancestor=cell_id, ancestor_birth=birth_time),
            by="ancestor") %>%
  mutate(weight=ifelse(is.na(birth_time - ancestor_birth), 0, birth_time - ancestor_birth))



# library(igraph)
# g = graph_from_data_frame(edges, directed=TRUE)
# E(g)$weight = edges$weight
# 
# root = setdiff(edges$ancestor, edges$cell_id)
# root_dists = distances(g, v=root, to=V(g), weights=E(g)$weight)
# 
# patristic_distance = function(a, b, g, root_dists) {
#   # get all ancestors of each
#   anc_a = subcomponent(g, a, mode="in")
#   anc_b = subcomponent(g, b, mode="in")
#   
#   common = intersect(names(anc_a), names(anc_b))
#   
#   if (length(common) == 0) return(NA)
#   mrca = common[which.max(root_dists[1, common])]
#   
#   # patristic distance = d(root,a) + d(root,b) - 2*d(root,mrca)
#   da = root_dists[1, a]
#   db = root_dists[1, b]
#   dm = root_dists[1, mrca]
#   
#   return(da + db - 2 * dm)
# }
# 
# cell_ids = nodes$cell_id
# n = length(cell_ids)
# patristic_mat = matrix(0, n, n, dimnames=list(cell_ids, cell_ids))
# 
# for (i in seq_len(n)) {
#   for (j in seq_len(n)) {
#     if (i < j) {
#       d = patristic_distance(cell_ids[i], cell_ids[j], g, root_dists)
#       # if (is.na(d)) {print(i); print(j)}
#       patristic_mat[i, j] = patristic_mat[j, i] = d
#     }
#   }
# }
# 
# tokeep_cells = as.character(unique(mutations$cell_id))
# patristic_mat[!is.na(patristic_mat)] = 0
# heatmap(patristic_mat[tokeep_cells, tokeep_cells])




