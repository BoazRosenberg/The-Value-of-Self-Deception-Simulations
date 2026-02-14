library(readr)
library(tidyverse)
source("setup.R")

#### Load data ####

sim_A = data.frame()

for (vf_set in  c("increasing", "diminishing")){
  
file_name = paste0("simulation_A_", vf_set,".csv")
temp_summ = read_csv(file.path(data_dir,file_name)) %>% 
  filter(epoch== max(epoch)) %>%
  mutate(EV_centered = EV - mean(EV[abs(bias-0.5) <0.01])) %>%
  group_by(bias) %>%
  summarise(EV_centered =mean(EV_centered)) %>%
  mutate(set = vf_set)

sim_A = bind_rows(sim_A, temp_summ)

}

####   Plot   ####

{
  g = ggplot(sim_A,aes( x= bias, y = EV_centered, fill = EV_centered))+
  geom_col()+
    geom_vline(xintercept =0.5, linetype = 1, linewidth = 18, col = "gray")+
    geom_hline(yintercept =0, linetype = 1)+
    scale_fill_viridis_c(option = "D") +  # viridis continuous scale
      custom_theme()+
    theme(
      legend.position = "none",
      panel.border = element_rect(
        colour = "black",
        fill = NA,
        linewidth = 0.6
      ))+
  labs(x = "Tau",
       y = "Perceived value")+
    facet_grid(set~.)
  
  plot(g)
  } # plot

ggsave(filename = "plots/sim_A.png", plot = g,
       width = 2000, height = 1500, units = "px", dpi = 300)


