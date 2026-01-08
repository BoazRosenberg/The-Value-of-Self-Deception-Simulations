source("setup.R")
library(mgcv)
library(ggplot2)
library(mgcv)
library(viridis)

{

sim3 = read_csv(file.path(data_dir,"simulation_C.csv")) %>%
  filter(epoch %in% c(1:3,5, max(epoch)))


sim3summ = sim3 %>% group_by(bias,temp_bias,epoch) %>%
  summarise(base_EV = mean(EV),
            action  = mean(action),
            cost = mean(C)) %>% 
  mutate(EV = base_EV - cost)
} # load data and summarize

{
  
save_dir = "plots/Simulation C/"
  
{

  n_grid  <- 200
  n_bins_ev   <- 10     # EV posterization
  n_bins_cost <- 10     # cost posterization
  k_smooth <- c(15, 15)
  
  width = 4
  height = 3
  dpi = 300
  
} # Global settings

{
  ev_range <- range(sim3summ$EV, na.rm = TRUE)
  
  ev_breaks <- seq(
    from = ev_range[1],
    to   = ev_range[2],
    length.out = n_bins_ev + 1
  )
  
  ev_mid <- (head(ev_breaks, -1) + tail(ev_breaks, -1)) / 2
  
  ev_colors <- viridis(n_bins_ev)
  
} # Value scale
  
{
  make_grid <- function(df, x, y, n = 200) {
    grid <- expand.grid(
      x = seq(min(df[[x]]), max(df[[x]]), length.out = n),
      y = seq(min(df[[y]]), max(df[[y]]), length.out = n)
    )
    names(grid) <- c(x, y)
    grid
  }
  
} # Create grid
  


} # Setup 

{
for (i in c(1, 2,3, max(sim3summ$epoch))) {
    
  {
    df_i <- sim3summ |> filter(epoch == i)
    
    gam_ev <- gam(EV ~ te(temp_bias, bias, k = k_smooth),
                  data = df_i)
    
    grid_ev <- make_grid(df_i, "temp_bias", "bias", n_grid)
    grid_ev$EV <- predict(gam_ev, newdata = grid_ev)
    
    grid_ev$EV_fill <- cut(grid_ev$EV, breaks = ev_breaks, include.lowest = TRUE)
    levels(grid_ev$EV_fill) <- ev_mid
    grid_ev$EV_fill <- as.numeric(as.character(grid_ev$EV_fill))
    
    } # Smooth
    
  g_ev <- ggplot(grid_ev, 
                 aes(x = temp_bias, y = bias, z = EV)) +
    
      geom_contour_filled(breaks = ev_breaks) +                             # Plot
      
      scale_fill_manual(values = ev_colors) +
      labs(x = expression(tau[temp]),
           y = expression(tau[cons]),
           fill = "Learnt Value") +
      custom_theme() +
      coord_fixed(expand = FALSE)+
      theme(legend.position = "none")
    
    ggsave(
      filename = paste0(save_dir,"value", i, ".png"),
      plot = g_ev,
      width = width,
      height = height,
      dpi = dpi
    )
  }
} # Value plots 

{
  
  for (i in c(3, 5, max(sim3summ$epoch))) {
    
    df_i <- sim3summ %>% filter(epoch == i)
    
    cost_range <- range(df_i$cost, na.rm = TRUE)
    
    cost_breaks <- seq(from = cost_range[1],
                       to   = cost_range[2],
                       length.out = n_bins_cost + 1)
    
    cost_colors <- colorRampPalette(c("#fff5f0", "#b2182b"))(n_bins_cost)
    
    gam_cost <- gam(cost ~ te(temp_bias, bias, k = k_smooth),
                    data = df_i)
    
    grid_cost <- make_grid(df_i, "temp_bias", "bias", n_grid)
    grid_cost$cost <- predict(gam_cost, newdata = grid_cost)
    
    g_cost <- ggplot(grid_cost,
                     aes(x = temp_bias, y = bias, z = cost)) +
      geom_contour_filled(breaks = cost_breaks) +
      scale_fill_manual(values = cost_colors, drop = FALSE) +
      labs(x = expression(tau[temp]),
           y = expression(tau[cons]),
           fill = "Cost") +
      custom_theme() +
      coord_fixed(expand = FALSE)+
      theme(legend.position = "none")
    
    ggsave(filename = paste0(save_dir,"cost", i, ".png"),
           plot = g_cost,
           width = width,
           height = height,
           dpi = dpi)
    
  }
  
} # Cost plots 

{ 
  action_levels <- floor(min(sim3summ$action)) : ceiling(max(sim3summ$action))
  action_colors <- colorRampPalette(c("#F1DFF2", "#7a2e80"))(length(action_levels))
  
  for (i in 1:3) {
    
   {
      df_i <- sim3summ |> filter(epoch == i)
      
      gam_action <- gam(action ~ te(temp_bias, bias, k = k_smooth),
                    data = df_i)
      
      grid_action <- make_grid(df_i, "temp_bias", "bias", n_grid)
      grid_action$action <- predict(gam_action, newdata = grid_action)
      
    } # Smooth
    
    
   g_pol <- ggplot( grid_action,
                    aes(x = temp_bias, y = bias, z = action)) +
      
      geom_contour_filled(breaks = action_levels + 0.5) +         # Plot
     
      scale_fill_manual(values = action_colors) +                # Colors 
      labs( x = expression(tau[temp]),                            # Labels
            y = expression(tau[cons]),
            fill = "Action") +
      custom_theme() +                                            # Theme
      coord_fixed(expand = FALSE)+
     theme(legend.position = "none") 
    
   # Save plot
    ggsave(filename = paste0(save_dir,"action", i, ".png"),
           plot = g_pol,
           width = width,
           height = height,
           dpi = dpi)
    
    
  }
  
} # Action plots

{
  {
    ev_bar = ggplot(data.frame(EV = ev_range),
                     aes(x = 1, y = EV, fill = EV) ) +
      geom_tile() +
      scale_fill_gradientn(colors = ev_colors,
                           limits = ev_range,
                           guide = guide_colorbar(barheight = unit(4, "cm"),
                                                  ticks = TRUE)) +
      theme_void() +
      theme(legend.position = "right",
            egend.title = element_text(),
            legend.text = element_text()
      ) +
      labs(fill = "Learnt Value")
    
    
    ggsave(
      filename = paste0(save_dir, "value_colorbar.png"),
      plot = ev_bar,
      width = 1.2,
      height = 3,
      dpi = dpi
    )
    
  } # Value
  
  {
    cost_range <- range(sim3summ$cost, na.rm = TRUE)
    
    cost_colors <- colorRampPalette(c("#fff5f0", "#b2182b"))(n_bins_cost)
    
    cost_bar <- ggplot(
      data.frame(cost = cost_range),
      aes(x = 1, y = cost, fill = cost)
    ) +
      geom_tile() +
      scale_fill_gradientn(
        colors = cost_colors,
        limits = cost_range,
        guide = guide_colorbar(
          barheight = unit(4, "cm"),
          ticks = TRUE
        )
      ) +
      theme_void() +
      labs(fill = "Mean Cost")
    
    ggsave(
      filename = paste0(save_dir, "cost_colorbar.png"),
      plot = cost_bar,
      width = 1.2,
      height = 3,
      dpi = dpi
    )
    
  } # Cost
  
  {
    action_bar_df <- data.frame( action = factor(action_levels, levels = action_levels),
                                 y = action_levels)
    action_bar <- ggplot(action_bar_df,
                         aes(x = 1, y = y, fill = action)) +
      geom_tile() +
      scale_fill_manual(
        values = action_colors,
        breaks = as.character(action_levels),
        guide = guide_legend(
          title = "Selected Action",
          title.position = "top",
          label.position = "right",
          keyheight = unit(0.5, "cm"),
          reverse = TRUE
        )
      ) +
      theme_void() +
      theme(
        legend.position = "right",
        legend.title = element_text(),
        legend.text = element_text()
      )
    
    ggsave(
      filename = paste0(save_dir, "action_colorbar.png"),
      plot = action_bar,
      width = 1.2,
      height = 3,
      dpi = dpi
    )
    
    
  } # Actions
  
} # Color bars
