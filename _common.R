knitr::opts_chunk$set(
  comment = "#>",
  message = FALSE, 
  warning = FALSE, 
  cache = TRUE, 
  eval = TRUE,
  tidy = "styler", 
  fig.width = 8, 
  fig.height = 5
  )

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
