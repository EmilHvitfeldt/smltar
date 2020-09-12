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
