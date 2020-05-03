# Contributing to _Supervised Machine Learning for Text Analysis in R_

**Thanks for considering contributing to our book!** 👍

Here are some helpful hints in getting the code in the book 📖 to run:

## spaCy installation

If you have both Anaconda and miniconda installed on your computer, you may have trouble installing and using the `"spacyr"` engine. To successfully install spaCy into an environment that R can find and use, first create a miniconda environment and then install spaCy into it.

```r
reticulate::conda_create("r-spacyr", 
                         conda = "/path/to/Library/r-miniconda/bin/python", 
                         packages = "python==3.6.9")
spacyr::spacy_install(conda = "/path/to/Library/r-miniconda/bin/python", 
                      envname = "r-spacyr")
```

Then, include a line such as this in any script using the `"spacyr"` engine for tokenization or text handling:

```r
spacyr::spacy_initialize(condaenv = "r-spacyr")
```
