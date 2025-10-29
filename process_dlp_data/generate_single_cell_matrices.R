# library(ProCESS)
library(dplyr)

main_path = "/orfeo/cephfs/scratch/cdslab/ebusca00/GitHub/snv_caller/process_dlp_data/high_cov_1500x/"

phylo_forest = load_phylogenetic_forest("/orfeo/cephfs/scratch/cdslab/ggandolfi/Github/ProCESS-DLP/dlp_simulation_1/phylo_forest_1.sff")

chromosomes = phylo_forest$get_absolute_chromosome_positions()$chr
seq_res = readRDS(file.path(main_path, "sequencing_high_cov.Rds"))

somatic_dp = lapply(seq_along(chromosomes), function(x){
  seq_res[[x]]$mutations %>% 
    filter(classes!="germinal") %>% 
    mutate(mutationID=paste0("chr",chr,":",from,":",ref,":",alt)) %>% 
    select(mutationID,sample,DP)
}) %>% bind_rows()

somatic_nv = lapply(seq_along(chromosomes), function(x){
  seq_res[[x]]$mutations %>% 
    filter(classes!="germinal") %>% 
    mutate(mutationID=paste0("chr",chr,":",from,":",ref,":",alt)) %>% 
    select(mutationID,sample,NV)
}) %>% bind_rows()


saveRDS(object=somatic_nv, file=file.path(main_path, "somatic_nv_high_cov.rds"))
saveRDS(object=somatic_dp, file=file.path(main_path, "somatic_dp_high_cov.rds"))
