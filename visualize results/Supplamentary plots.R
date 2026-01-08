library(readr)
library(tidyverse)
source("setup.R")


df_dir <- normalizePath(file.path("..", "..", "..", "Simulation", "python", "self-deception",
                                "estimate learning processes", "S_hat.csv"))

S_hat <- read_csv(df_dir)

copies = 100
{
ggplot(S_hat %>% filter(copy <copies ), aes(x = tau, y = Q))+
  geom_line(aes(group = interaction(eta,copy), col = eta), alpha = 0.1)+
  scale_color_viridis_c(option = "C") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = expression(tau),
    y = "Q",
    color = expression(eta)
  )
} # by eta - lines

bins = 5
smoothness = 100
{
  library(ggplot2)
  library(dplyr)
  library(viridis)
  
  ggplot(
    S_hat %>% filter(copy < copies),
    aes(x = tau, y = Q)
  ) +
    stat_density_2d(
      aes(fill = after_stat(density)),
      geom = "raster",
      contour = FALSE,
      n = smoothness
    ) +
    scale_fill_viridis_c(
      option = "C",
      #trans = "sqrt"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      legend.text = element_text(color = "black"),
      legend.title = element_text(color = "black")
    ) +
  labs(
  x = expression(tau),
  y = expression(hat(Z)),
  fill = "Density"
)
} # heatmap

n_eta_cat = 4
{
  library(ggplot2)
  library(dplyr)
  library(viridis)
  
  ggplot(
    S_hat %>% filter(copy < copies)%>%
      mutate(eta_cat = cut(eta,n_eta_cat)),
    aes(x = tau, y = Q)
  ) +
    stat_density_2d(
      aes(fill = after_stat(density)),
      geom = "raster",
      contour = FALSE,
      n = smoothness
    ) +
    scale_fill_viridis_c(
      option = "C",
      #trans = "sqrt"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      legend.text = element_text(color = "black"),
      legend.title = element_text(color = "black")
    ) +
    labs(
      x = expression(tau),
      y = "Q",
      fill = "Density"
    )+
    facet_wrap(~ eta_cat)
  
} # heatmap
