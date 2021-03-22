library(knitr)
opts_chunk$set(
  comment = "#>",
  message = FALSE, 
  warning = FALSE, 
  cache = TRUE, 
  eval = TRUE,
  tidy = "styler", 
  dpi = 105, # this creates 2*105 dpi at 6in, which is 300 dpi at 4.2in
  fig.align = 'center',
  fig.width = 6,
  fig.asp = 0.618  # 1 / phi
)

opts_template$set(
  fig.large = list(fig.asp = 0.8),
  fig.square = list(fig.asp = 1),
  fig.long = list(fig.asp = 1.5)
)

# library(paletteer)
# library(prismatic)
# library(magrittr)
# paletteer_d("RColorBrewer::Set3") %>%
#   clr_saturate(0.25) %>%
#   clr_darken(0.15) %>%
#   plot()
discrete_colors <- c("#5BBCACFF", "#D5D587FF", "#9993C5FF", "#DE6454FF", 
                     "#5497C2FF", "#DA9437FF", "#92C22BFF", "#D8A8C1FF", 
                     "#C0ACACFF", "#B556B7FF", "#A3CA9AFF", "#D7C637FF")

alpha_viridis <- function(...) {
  scale_fill_gradientn(..., colors = viridis::viridis(256, alpha = 0.7))
}

suppressPackageStartupMessages(library(tidyverse))

theme_set(theme_light())

update_geom_defaults("col", list(fill = "#8097ae", alpha = 0.9))
update_geom_defaults("point", list(color = "#566675"))
update_geom_defaults("line", list(color = "#566675", alpha = 0.7))

options(
  ggplot2.discrete.fill = discrete_colors,
  ggplot2.discrete.colour = discrete_colors,
  ggplot2.continuous.fill = alpha_viridis,
  ggplot2.continuous.colour = alpha_viridis
)

# https://github.com/EmilHvitfeldt/smltar/issues/114
hook_output = knit_hooks$get('output')
knit_hooks$set(output = function(x, options) {
  # this hook is used only when the linewidth option is not NULL
  if (!is.null(n <- options$linewidth)) {
    x = knitr:::split_lines(x)
    # any lines wider than n should be wrapped
    if (any(nchar(x) > n)) x = strwrap(x, width = n)
    x = paste(x, collapse = '\n')
  }
  hook_output(x, options)
})

options(crayon.enabled = FALSE)

library(htmltools)
library(quanteda)


columnize <- function(words, ncol = 5) {
  
  tagList(
    tags$div(
      words %>%
        map(tags$p) %>%
        tagList(),
      style = sprintf("column-count:%d;font-size:11pt;line-height:11.5pt", 
                      as.integer(ncol))
    )
  )
  
}

## control caching
online <- TRUE
# online <- FALSE

sparse_bp <- hardhat::default_recipe_blueprint(composition = "dgCMatrix")

## for Keras chapters

keras_predict <- function(model, baked_data, response) {
  predictions <- predict(model, baked_data)[, 1]
  
  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 < 0.5, 0, 1),
    state = response
  ) %>%
    mutate(across(c(state, .pred_class),
                  ~ factor(.x, levels = c(1, 0))))
}

autoplot.conf_mat <- function(object, type = "heatmap", ...) {
  cm_heat(object)
}

cm_heat <- function(x) {
  `%+%` <- ggplot2::`%+%`
  
  table <- x$table
  
  df <- as.data.frame.table(table)
  
  # Force known column names, assuming that predictions are on the
  # left hand side of the table (#157).
  names(df) <- c("Prediction", "Truth", "Freq")
  
  # Have prediction levels going from high to low so they plot in an
  # order that matches the LHS of the confusion matrix
  lvls <- levels(df$Prediction)
  df$Prediction <- factor(df$Prediction, levels = rev(lvls))
  
  df %>%
    ggplot2::ggplot(
      ggplot2::aes(
        x = Truth,
        y = Prediction,
        fill = Freq
      )
    ) %+%
    ggplot2::geom_tile() %+%
    ggplot2::scale_fill_gradient(low = "grey99", high = "#5781AE") %+%
    ggplot2::theme(
      panel.background = ggplot2::element_blank(),
      legend.position = "none"
    ) %+%
    ggplot2::geom_text(ggplot2::aes(label = Freq))
}


tidy.LiblineaR <- function(x, ...) {
  
  ret <- tibble(colnames(x$W), x$W[1,])
  colnames(ret) <- c("term", "estimate")
  
  ret
}

conf_mat_resampled <- function(x) {
  tune::conf_mat_resampled(x, tidy = FALSE) %>%
    as.table() %>%
    conf_mat()
}
