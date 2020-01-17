library(tidyverse)

read_csv("data-raw/complaints.csv") %>%
  janitor::clean_names() %>%
  filter(date_received >= "2019-01-01",
         !is.na(consumer_complaint_narrative)) %>%
  write_csv("data/complaints.csv.gz")
