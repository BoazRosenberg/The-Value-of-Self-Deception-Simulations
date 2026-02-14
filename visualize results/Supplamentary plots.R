library(readr)
library(tidyverse)
source("setup.R")

#### Visualize the distribution of learning rate values used in S_hat.csv ####

x = seq(0,1,0.001)

a = 2
b = 10

df_beta = data.frame(x,
                     d = dbeta(x,a,b))

{
  g_lr_dist = ggplot(df_beta, aes(x = x, y = d)) +
    geom_area(fill = "skyblue", alpha = 0.6) +          # filled area with transparency
    geom_line(color = "steelblue", size = 1) +          # smooth line on top
    labs(
      title = "Distribution of Agent Learning Rate (η)",
      subtitle = "Used in the approximation of agents' belief regarding their state",
      x = expression(eta),
      y = "Density"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      axis.line.y = element_blank(),    # remove y-axis line
      axis.ticks.y = element_blank(),   # remove y-axis ticks
      axis.text.y = element_blank(),    # remove y-axis labels
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold")
    )
  
  plot(g_lr_dist)
  
  ggsave("plots/Supplamentary LR distribution.png",g_lr_dist)
  
}



#### Heatmap of learnt S hat values for each value of tau ####

#  ** Heavy plot **  may take a while

# Represents S hat values for observations ~ N(0,1)

S_hat <- read_csv(S_hat_dir)

smoothness = 100
{
  library(ggplot2)
  library(dplyr)
  library(viridis)
  
  g_S_hat = ggplot(S_hat,aes(x = tau, y = Q)) +
    stat_density_2d(aes(fill = after_stat(density)),
                    geom = "raster",
                    contour = FALSE,
                    n = smoothness) +
    scale_fill_viridis_c(option = "C",) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid = element_blank(),
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      legend.text = element_text(color = "black"),
      legend.title = element_text(color = "black")
    ) +
    labs(x = expression(tau),
         y = expression(hat(Z)),
         fill = "Density")
  
  
  ggsave("plots/Supplamentary S hat heatmap.png",g_S_hat)
} # heatmap
