library(tidyverse)

data_folder = "process_dlp_data/high_cov_20x/out/"

mutations = readRDS("process_dlp_data/mutations.rds") %>% as_tibble() %>% 
  mutate(mutationID=paste(chr, from, ref, alt, sep=":")) %>%
  select(cell_id, mutationID, type, cause, class, allele)

readRDS("process_dlp_data/nodes.rds") %>% inner_join(y=muts_sequenced, by="sample")

compute_is.present = function(mutation_ids, cell_ids) {
  tmp = mutations %>% filter(mutationID==mutation_ids, cell_id==cell_ids)
  return(nrow(tmp) == 0)
}

data_true = read.csv(file.path(data_folder, "data_true.csv"), row.names=1) %>% as_tibble() %>% 
  mutate(mutation_ids=str_remove_all(mutation_ids, "chr")) %>% 
  left_join(
    mutations %>% select(mutationID, cell_id) %>% mutate(found=TRUE),
    by=c("mutation_ids"="mutationID", "cell_ids"="cell_id")
  ) %>%
  mutate(is.present=!is.na(found)) %>% select(-found) %>% 
  mutate(VAF_true=ifelse(is.present, 0.5, 0))

mutation_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% pull(mutation_ids)
cell_ids = data_true %>% select(mutation_ids, cell_ids, VAF) %>% 
  pivot_wider(names_from="cell_ids", values_from="VAF") %>% select(-mutation_ids) %>% 
  colnames

data_true %>% 
  ggplot() +
  geom_point(aes(x=VAF_true, y=AD))

data_true %>% 
  ggplot() +
  geom_point(aes(x=VAF, y=AD, color=clone_ids))


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

logit = function(p) {
  log(p / (1 - p))
}
expit = function(x) {
  exp(x) / (1 + exp(x))
}

# heatmap(wout_kernel$theta %>% as.matrix())
# heatmap(w_kernel$theta %>% as.matrix())

final_vafs = w_kernel$theta %>% tibble::rownames_to_column("mutation_ids") %>% 
  pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
  mutate(type="w_kernel") %>% 
  
  bind_rows(
    wout_kernel$theta %>% tibble::rownames_to_column("mutation_ids") %>% 
      pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
      mutate(type="wout_kernel") 
  ) %>% 
  mutate(cell_ids=as.integer(cell_ids))

final_vafs %>% 
  left_join(data_true) %>% 
  ggplot() +
  geom_boxplot(aes(x=is.present, y=VAF_inf, color=type), position=position_dodge(), outliers=FALSE) +
  ggbeeswarm::geom_beeswarm(aes(x=is.present, y=VAF_inf, color=type)) +
  facet_wrap(~is.present, scales="free")

data_true %>% 
  ggplot() +
  geom_boxplot(aes(x=factor(VAF_true), y=VAF), position=position_dodge())


final_vafs %>% 
  left_join(data_true) %>% 
  ggplot() +
  geom_point(aes(x=VAF, y=VAF_inf, color=type)) +
  geom_abline()
# facet_grid(~type)


input_d = final_vafs %>% 
  left_join(data_true) %>% 
  select(mutation_ids, cell_ids, VAF_inf, VAF, type, is.present) %>% 
  reshape2::melt(id=c("mutation_ids", "cell_ids", "is.present", "type")) %>% as_tibble() %>% 
  mutate(variable=factor(variable, levels=c("VAF","VAF_inf")))

input_d %>% 
  
  # filter(type=="w_kernel", is.present) %>% 
  
  ggplot() +
  # geom_point(aes(x=variable, y=value), size=0.5) +
  # geom_line(aes(x=variable, y=value, group=interaction(mutation_ids, cell_ids))) +
  geom_violin(aes(x=variable, y=value), draw_quantiles=c(0.5)) +
  facet_grid(type ~ is.present)


input_d %>% 
  ggplot() +
  geom_point(aes(x=VAF, y=VAF_inf), size=1, alpha=0.5) +
  geom_abline() +
  facet_grid(type ~ is.present) +
  theme_bw()


input_d %>% filter(!is.present)

data_true %>% filter(!is.present) %>% 
  filter(VAF > 0.1)


input_d %>% 
  filter(value > 0.001) %>% 
  ggplot() +
  geom_density(aes(x=value, fill=variable), position="identity", alpha=.5) +
  facet_grid(type ~ is.present) + theme_bw()



