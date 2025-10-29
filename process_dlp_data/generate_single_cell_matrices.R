library(ProCESS)
library(dplyr)

phylo_forest <- load_phylogenetic_forest("/orfeo/cephfs/scratch/cdslab/ggandolfi/Github/ProCESS-DLP/dlp_simulation_1/phylo_forest_1.sff")
sample_forest <- load_sample_forest("/orfeo/cephfs/scratch/cdslab/ggandolfi/Github/ProCESS-DLP/dlp_simulation_1/sample_forest_1.sff")

chromosomes <- phylo_forest$get_absolute_chromosome_positions()$chr
seq_res <- readRDS("sequencing_high_cov.Rds") ### seq res
somatic_dp <- lapply(seq_along(chromosomes), function(x){
  seq_res[[x]]$mutations %>% 
    filter(classes!="germinal") %>% 
    mutate(mutationID=paste0("chr",chr,":",from,":",ref,":",alt)) %>% 
    select(mutationID,sample,DP)
}) %>% bind_rows()

somatic_nv <- lapply(seq_along(chromosomes), function(x){
  seq_res[[x]]$mutations %>% 
    filter(classes!="germinal") %>% 
    mutate(mutationID=paste0("chr",chr,":",from,":",ref,":",alt)) %>% 
    select(mutationID,sample,NV)
}) %>% bind_rows()


saveRDS(object = somatic_nv,file = "somatic_nv_high_cov.rds")
saveRDS(object = somatic_dp,file = "somatic_dp_high_cov.rds")
