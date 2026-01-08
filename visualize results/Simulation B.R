library(readr)
library(ggplot2)
library(viridis)
library(mgcv)
source("setup.R")

sim_B = read_csv(file.path(data_dir,"simulation_B.csv"))

{
  smoothing_factor = 3
  # Fit a GAM surface to EV
  gam_fit <- mgcv::gam(
    EV ~ te(observation_noise, bias, k = c(smoothing_factor, smoothing_factor)),
    data = sim_B
  )
  
  # Predict on a fine grid
  grid <- expand.grid(
    observation_noise = seq(min(sim_B$observation_noise), max(sim_B$observation_noise), length.out = 200),
    bias = seq(min(sim_B$bias), max(sim_B$bias), length.out = 200)
  )
  grid$EV <- predict(gam_fit, newdata = grid)
} # Smooth results
  
{
 g =  ggplot(grid, aes(x = observation_noise, y = bias, z = EV)) +
    geom_contour_filled(bins = 10) +
    scale_fill_viridis_d(option = "viridis") +
    labs(
      x = expression(sigma),
      y = expression(tau),
      fill = "EV"
    ) +
    custom_theme() +
    coord_fixed(expand = FALSE) +
    theme(aspect.ratio = 1)
 
 plot(g)
  
}# Plot

ggsave(
  filename = paste0("plots/sim_B.png"),
  plot = g,
  width = 8, height = 6, dpi = 600)







