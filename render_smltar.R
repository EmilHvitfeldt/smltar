## assume up-to-date version of R and pandoc

##----------------------------------------------------------

devtools::install_dev_deps()

##----------------------------------------------------------

## no longer doing this on ARM:
## assume Miniconda installed and R environment exists called `r-reticulate`:
# reticulate::install_miniconda()
# reticulate::conda_create('r-reticulate', python_version = "3.6.9")
# keras::install_keras(tensorflow = '2.2', extra_packages = c('IPython', 'requests', 'certifi', 'urllib3'))

spacyr::spacy_install(envname = "tf_env", prompt = FALSE)

##----------------------------------------------------------

## double check for unexpected GitHub versions (only expect scotus from GH):
deps <- desc::desc_get_deps()
pkgs <- sort(deps$package[deps$type == "Imports"])
sessioninfo::package_info(pkgs, dependencies = FALSE)

##----------------------------------------------------------

# DELETE _tuning dir THEN

bookdown::render_book("index.Rmd", quiet = FALSE)

##----------------------------------------------------------

