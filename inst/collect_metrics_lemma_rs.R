library(tidyverse)
library(tidymodels)
library(textrecipes)
library(scotus)

spacyr::spacy_initialize(entity = FALSE)

set.seed(1234)
scotus_split <- scotus_filtered %>%
  mutate(year = as.numeric(year),
         text = str_remove_all(text, "'")) %>%
  initial_split()

scotus_train <- training(scotus_split)

set.seed(123)
scotus_folds <- vfold_cv(scotus_train)

svm_spec <- svm_linear() %>%
  set_mode("regression") %>%
  set_engine("LiblineaR")

svm_wf <- workflow() %>%
  add_model(svm_spec)

lemma_rec <- recipe(year ~ text, data = scotus_train) %>%
  step_tokenize(text, engine = "spacyr") %>%
  step_lemma(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) %>%
  step_normalize(all_predictors())

lemma_rs <- fit_resamples(
  svm_wf %>% add_recipe(lemma_rec),
  scotus_folds,
  control = control_resamples(verbose = TRUE)
)

collect_metrics_lemma_rs <- collect_metrics(lemma_rs)

write_csv(collect_metrics_lemma_rs, "inst/collect_metrics_lemma_rs.csv")