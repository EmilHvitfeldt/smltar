library(tidyverse)

read_csv("data-raw/complaints.csv") %>%
  filter(`Date received` >= "2019-01-01") %>%
  write_csv("data/complaints.csv.gz")
