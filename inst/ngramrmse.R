library(tidyverse)
library(scotus)

library(tidymodels)
library(textrecipes)
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

ngram_rec <- function(ngram_options) {
  recipe(year ~ text, data = scotus_train) %>%
    step_tokenize(text, token = "ngrams", options = ngram_options) %>%
    step_tokenfilter(text, max_tokens = 1e3) %>%
    step_tfidf(text) %>%
    step_normalize(all_predictors())
}

fit_ngram <- function(ngram_options) {
  fit_resamples(
    svm_wf %>% add_recipe(ngram_rec(ngram_options)),
    scotus_folds
  )
}

set.seed(123)
unigram_rs <- fit_ngram(list(n = 1))

set.seed(234)
bigram_rs <- fit_ngram(list(n = 2, n_min = 1))

set.seed(345)
trigram_rs <- fit_ngram(list(n = 3, n_min = 1))

collect_metrics_bigram_rs <- collect_metrics(bigram_rs)

write_csv(collect_metrics_bigram_rs, "inst/collect_metrics_bigram_rs.csv")

collect_metrics_all_ngram <- list(`1` = unigram_rs,
                                  `1 and 2` = bigram_rs,
                                  `1, 2, and 3` = trigram_rs) %>%
  map_dfr(collect_metrics, .id = "name") %>%
  filter(.metric == "rmse")

write_csv(collect_metrics_all_ngram, "inst/collect_metrics_all_ngram.csv")
