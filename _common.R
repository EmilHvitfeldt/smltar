library(knitr)
opts_chunk$set(
  comment = "#>",
  message = FALSE, 
  warning = FALSE, 
  cache = TRUE, 
  eval = TRUE,
  tidy = "styler", 
  fig.width = 8, 
  fig.height = 5
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

suppressPackageStartupMessages(library(tidyverse))
theme_set(theme_light())

library(htmltools)
library(quanteda)

update_geom_defaults("col", list(fill = "#8097ae", alpha = 0.9))
update_geom_defaults("point", list(color = "#566675"))
update_geom_defaults("line", list(color = "#566675", alpha = 0.7))


columnize <- function(words, ncol = 5,
                      style = "p { font-family:'Cabin', sans-serif;font-size:11pt;line-height:11.5pt;padding:0;margin:0}") {
  
  tagList(
    tags$style(style),
    tags$div(
      words %>%
        map(tags$p) %>%
        tagList(),
      style = sprintf("column-count:%d", as.integer(ncol))
    )
  )
  
}

## control caching
online <- TRUE
# online <- FALSE

sparse_bp <- hardhat::default_recipe_blueprint(composition = "dgCMatrix")

## for Keras chapters
library(keras)
tensorflow::tf$random$set_seed(1234)

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
