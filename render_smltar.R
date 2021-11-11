## assume up-to-date version of R and pandoc
## update renv if needed

##----------------------------------------------------------

## for Julia's ARM mac:
options(reticulate.conda_binary = path.expand("~/miniforge3/bin/conda"))
spacyr::spacy_initialize(condaenv = "tf_env", entity = FALSE)
spacyr::spacy_initialize(entity = FALSE)

##----------------------------------------------------------

## not really necessary because of renv, but if desired
## double check for unexpected GitHub versions (only expect scotus from GH):
deps <- desc::desc_get_deps()
pkgs <- sort(deps$package[deps$type == "Imports"])
sessioninfo::package_info(pkgs, dependencies = FALSE)

##----------------------------------------------------------

bookdown::render_book("index.Rmd", quiet = FALSE)

##----------------------------------------------------------

