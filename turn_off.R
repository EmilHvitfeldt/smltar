turn_off <- function(x) {
  text <- readLines(x)
  text <- gsub(pattern = ", eval=\\!?online", replace = "", x = text)
  text <- gsub(pattern = "`r ", replace = "`r #", x = text)
  text <- gsub(pattern = "`r #if \\(", replace = "`r if \\(", x = text)
  text <- gsub(pattern = "eval ?= ?TRUE", replace = "eval=FALSE", x = text)
  text <- gsub(pattern = r"(```\{r, eval=!?knitr:::is_html_output\(\))", replacement = "```{r, eval=FALSE", x = text)
  text <- gsub(pattern = r"(echo=knitr:::is_html_output\(\))", replacement = "echo=FALSE", x = text)
  writeLines(text, x)
}

purrr::walk(fs::dir_ls(regexp = "[0-9]+.*\\.Rmd$"), turn_off)
writeLines(
  '
    knitr::opts_chunk$set(
  comment = "#>",
    message = FALSE, 
    warning = FALSE, 
    cache = TRUE, 
    eval = FALSE,
    tidy = "styler", 
    fig.align = "center",
    fig.width = 6,
    fig.asp = 0.618  # 1 / phi
  )
    ',
  "_common.R"
)

single_core <- function(x) {
  text <- readLines(x)
  text <- gsub(pattern = "doParallel::registerDoParallel()", replace = "#doParallel::registerDoParallel()", x = text)
  writeLines(text, x)
}

#purrr::walk(fs::dir_ls(regexp = "[0-9]+.*\\.Rmd$"), single_core)