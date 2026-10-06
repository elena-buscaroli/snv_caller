library(tidyverse)

my_theme = theme_light(base_size=10) +
  theme(legend.position="bottom",
        legend.key.size=unit(0.3, "cm"),
        legend.key.spacing=unit(1, "mm"),
        panel.background=element_rect(fill="white"),
        axis.text.x=element_text(size=8),
        axis.text.y=element_text(size=8),
        axis.title=element_text(size=10),
        legend.text=element_text(size=8),
        legend.title=element_text(size=10),
        text=element_text(size=10), 
        plot.title=element_text(size=12),
        plot.tag=element_text(size=18, face="bold"))

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


data_inf = load_matrix(file.path(data_folder, "gamma_hat_kernel.csv"), mutation_ids, cell_ids) %>% 
  expit() %>% tibble::rownames_to_column("mutation_ids") %>% 
  pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
  mutate(type="Latent GP") %>% 
  mutate(found=ifelse(VAF_inf > 0.3, "Present", "Absent")) %>% 
  
  bind_rows(
    data_true %>% 
      rename(VAF_inf=VAF) %>% unique() %>% 
      mutate(type="Heuristics") %>% 
      mutate(found=ifelse(AD>=5 & DP>=5 & VAF_inf>=0.3, "Present", "Absent")) %>% 
      select(mutation_ids, cell_ids, VAF_inf, type, found)
  )
  
  # bind_rows(
  #   load_matrix(file.path(data_folder, "gamma_hat_Nkernel.csv"), mutation_ids, cell_ids) %>% 
  #     expit() %>% tibble::rownames_to_column("mutation_ids") %>% 
  #     pivot_longer(cols=-"mutation_ids", values_to="VAF_inf", names_to="cell_ids") %>% 
  #     mutate(type="wout_kernel") 
  # ) %>% 


merged_df = data_inf %>% 
  left_join(data_true %>% select(mutation_ids, cell_ids, VAF, presence, presence_dropout, AD, DP)) %>% 
  mutate(classification=ifelse(found=="Present" & presence=="Present", "TP",
                               ifelse(found=="Absent" & presence=="Absent", "TN",
                                      ifelse(found=="Present" & presence=="Absent", "FP", "FN"))))

pl_metrics = merged_df %>% 
  ggplot() +
  geom_bar(aes(x=classification, fill=type), position=position_dodge(width=0.6),
           width=0.5)

# merged_df %>% 
#   filter(classification=="TP") %>% 
#   mutate(ratio=VAF / VAF_inf) %>%
# 
#   ggplot() +
#   # stat_density_2d(aes(x=VAF, y=ratio, fill=type, alpha=..level..),
#   #                 geom="polygon", bins=50) +
#   geom_hline(yintercept=1, linetype="dashed", linewidth=0.5, color="grey80") +
#   geom_violin(aes(x=type, y=ratio), draw_quantiles=c(0.5))


# merged_df %>% 
#   filter(found=="Present") %>%
#   
#   ggplot(aes(x=1:length(mutation_ids))) +
#   geom_line(aes(y=cumsum(VAF)), color="black") +
#   geom_line(aes(y=cumsum(VAF_inf)), color="red")
# 
# merged_df %>% 
# 
#   ggplot(aes(x=1:length(mutation_ids))) +
#   geom_line(aes(y=cumsum(VAF)), color="black") +
#   geom_line(aes(y=cumsum(VAF_inf)), color="red")


library(dplyr)
library(ggplot2)

pl_cumsum_found = merged_df %>%
  filter(found == "Present") %>%
  arrange(VAF) %>%
  mutate(cum_VAF=cumsum(VAF),
         cum_VAF_inf=cumsum(VAF_inf)) %>%
  ggplot(aes(x=1:length(mutation_ids))) +
  geom_line(aes(y=cum_VAF, color="Observed"), size=1) +
  geom_line(aes(y=cum_VAF_inf, color="Inferred"), size=1)


pl_cumsum_true = merged_df %>%
  filter(presence == "Present") %>%
  arrange(VAF) %>%
  mutate(cum_VAF=cumsum(VAF),
         cum_VAF_inf=cumsum(VAF_inf)) %>%
  ggplot(aes(x=1:length(mutation_ids))) +
  geom_line(aes(y=cum_VAF, color="Observed"), size=1) +
  geom_line(aes(y=cum_VAF_inf, color="Inferred"), size=1)


# merged_df %>%
#   filter(presence_dropout == "Present") %>%
#   arrange(VAF) %>%
#   mutate(
#     cum_VAF = cumsum(VAF),
#     cum_VAF_inf = cumsum(VAF_inf)
#   ) %>%
#   ggplot(aes(x=1:length(mutation_ids))) +
#   geom_line(aes(y = cum_VAF), color = "black") +
#   geom_line(aes(y = cum_VAF_inf), color = "red") +
#   theme_bw()



# merged_df %>% 
#   filter(type=="w_kernel") %>% 
#   ggplot() + 
#   geom_boxplot(aes(x=classification, y=VAF_inf))
# 
# 
# merged_df %>% 
#   filter(type=="w_kernel", classification=="TP") %>% 
#   ggplot() + 
#   geom_point(aes(x=VAF, y=VAF_inf)) +
#   ylim(0,1) + xlim(0,1)





# Save ####

a = pl_metrics + my_theme +
  scale_fill_manual(values=c("steelblue", "goldenrod"), name="Model") +
  ylab("Count") + theme(axis.title.x=element_blank()) + labs(tag="a")

b = patchwork::wrap_plots(pl_cumsum_found + labs(tag="b"), pl_cumsum_true + labs(tag="c")) &
  my_theme &
  scale_color_manual(values=c("goldenrod", "forestgreen"), name="VAF") &
  scale_x_continuous(labels=scales::comma, name="Mutations",
                     breaks=seq(0, 200000, by=100000)) &
  ylab("Cumulative VAF") # & theme(axis.title.x=element_blank())


figure = patchwork::wrap_plots(a, b, widths=c(1,2)) &
  theme(panel.background=element_rect(fill="transparent"),
        plot.background=element_rect(fill="transparent", color=NA),
        legend.background=element_rect(fill="transparent"),
        legend.box.background=element_rect(fill="transparent", colour="transparent"))
ggsave("~/Dropbox/work/00_THESIS/figures/cellsnp/fig5.png", plot=figure, dpi=300, bg="transparent",
       width=21, height=8, units="cm", device=png, family="Helvetica")

