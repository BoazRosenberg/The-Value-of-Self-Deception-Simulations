library(tidyverse)


data_dir = normalizePath(file.path("..", "csv files"))


logit = function(x){
  return( log(x/(1-x)))
}

custom_theme <- function(base_size = 14, base_family = "") {
  
  theme_minimal(base_size = base_size, base_family = base_family) +
    
    theme(
      panel.grid = element_blank(),
      axis.ticks = element_blank(),
      legend.title = element_text(size = base_size - 1, face = "bold"),
      legend.text = element_text(size = base_size - 3),
      plot.title = element_text(size = base_size + 2, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = base_size, hjust = 0.5),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(size = base_size - 3),
      plot.caption = element_text(size = base_size - 4, hjust = 1, face = "italic"),
      panel.border = element_blank(),
      
      
    )
  }



