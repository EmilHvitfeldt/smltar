turn_off <- function(x) {
  text <- readLines(x)
  text <- gsub(pattern = ", eval=\\!?online", replace = "", x = text)
  text <- gsub(pattern = "`r ", replace = "`r #", x = text)
  text <- gsub(pattern = "eval ?= ?TRUE", replace = "eval=FALSE", x = text)
  writeLines(text, x)
}

purrr::walk(fs::dir_ls(regexp = "[0-9]+.*\\.Rmd$"), turn_off)
