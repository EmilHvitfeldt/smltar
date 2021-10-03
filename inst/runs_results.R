library(tidyverse)

kickstarter <- read_csv("data/kickstarter.csv.gz")
kickstarter

library(tidymodels)
set.seed(1234)
kickstarter_split <- kickstarter %>%
  filter(nchar(blurb) >= 15) %>%
  initial_split()

kickstarter_train <- training(kickstarter_split)
kickstarter_test <- testing(kickstarter_split)

library(textrecipes)

max_words <- 20000
max_length <- 30

kick_rec <- recipe(~blurb, data = kickstarter_train) %>%
  step_tokenize(blurb) %>%
  step_tokenfilter(blurb, max_tokens = max_words) %>%
  step_sequence_onehot(blurb, sequence_length = max_length)

set.seed(234)
kick_val <- validation_split(kickstarter_train, strata = state)

kick_prep <- prep(kick_rec)
kick_analysis <- bake(kick_prep, new_data = analysis(kick_val$splits[[1]]),
                      composition = "matrix")

kick_assess <- bake(kick_prep, new_data = assessment(kick_val$splits[[1]]),
                    composition = "matrix")

state_analysis <- analysis(kick_val$splits[[1]]) %>% pull(state)
state_assess <- assessment(kick_val$splits[[1]]) %>% pull(state)

julias_computer <- FALSE

library(keras)

if (julias_computer) {
  library(tensorflow)
  
  ## for Julia's ARM chip
  use_python("~/miniforge3/bin/python")
  use_condaenv("tf_env")
  reticulate::py_discover_config("tensorflow")
}
library(tensorflow)
tensorflow::tf$random$set_seed(1234)

junk <- keras_model_sequential()

hyperparams <- list(
  kernel_size1 = c(3, 5, 7),
  strides1 = c(1, 2)
)

library(tfruns)
runs <- tuning_run(
  file = "cnn-spec.R",
  runs_dir = "_tuning",
  flags = hyperparams
)

runs_results <- as_tibble(ls_runs())

write_csv(runs_results, "inst/runs_results.csv")

fs::dir_delete("_tuning")