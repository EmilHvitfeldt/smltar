### CFPB complaints

library(tidyverse)

read_csv("data-raw/complaints.csv") %>%
  janitor::clean_names() %>%
  filter(date_received >= "2019-01-01",
         !is.na(consumer_complaint_narrative)) %>%
  write_csv("data/complaints.csv.gz")

### DOJ press releases

library(jsonlite)
library(tidyverse)

out <- stream_in(file("data-raw/combined.json")) %>% 
  as_tibble() %>%
  mutate(agency = map_chr(components, `[`, 1),
         date = as.Date(date)) %>%
  select(date, agency, title, contents)

out %>%
  write_csv("data/press_releases.csv.gz")

### Kickstarter blurbs
# https://webrobots.io/kickstarter-datasets/

dir_ls(c("~/Downloads/Kickstarter_2016-01-28T09_15_08_781Z", 
         "~/Downloads/Kickstarter_2016-03-22T07_41_08_591Z/")) %>%
  map_dfr(read_csv, col_types = cols_only(blurb = col_character(),
                                          state = col_character())) %>%
  filter(state %in% c("failed", "successful")) %>%
  mutate(state = as.numeric(state == "successful")) %>%
  write_csv("data/kickstarter.csv.gz")
