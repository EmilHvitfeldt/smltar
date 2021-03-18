## assume up-to-date version of R and pandoc

##----------------------------------------------------------

install.packages("pak", repos = "https://r-lib.github.io/p/pak/dev/")
pak::local_install_dev_deps(upgrade = TRUE)

##----------------------------------------------------------

pak::pkg_install('rstudio/reticulate')

## assume Miniconda installed and R environment exists called `r-reticulate`:
# reticulate::install_miniconda()
# reticulate::conda_create('r-reticulate', packages = c('python==3.6.9'))

keras::install_keras(tensorflow = '2.2', extra_packages = c('IPython', 'requests', 'certifi', 'urllib3'))
spacyr::spacy_install(python_version = "3.6.9", envname = "spacy_condaenv", prompt = FALSE)

##----------------------------------------------------------

## double check for unexpected GitHub versions:
deps <- desc::desc_get_deps()
pkgs <- sort(deps$package[deps$type == "Imports"])
sessioninfo::package_info(pkgs, dependencies = FALSE)

##----------------------------------------------------------

bookdown::render_book("index.Rmd", quiet = FALSE)

##----------------------------------------------------------
