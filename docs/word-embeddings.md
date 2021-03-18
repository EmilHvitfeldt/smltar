# Word Embeddings {#embeddings}



> You shall know a word by the company it keeps.
> \hfill --- [John Rupert Firth](https://en.wikiquote.org/wiki/John_Rupert_Firth)

So far in our discussion of natural language features, we have discussed preprocessing steps such as tokenization, removing stop words, and stemming in detail. We implement these types of preprocessing steps to be able to represent our text data in some data structure that is a good fit for modeling. An example of such a data structure is a sparse matrix. Perhaps, if we wanted to analyse or build a model for consumer complaints to the [United States Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/), we would start with straightforward word counts.


```r
library(tidyverse)
library(tidytext)
library(SnowballC)

complaints <- read_csv("data/complaints.csv.gz")

complaints %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  anti_join(get_stopwords()) %>%
  mutate(stem = wordStem(word)) %>%
  count(complaint_id, stem) %>%
  cast_dfm(complaint_id, stem, n)
```

```
## Document-feature matrix of: 117,214 documents, 46,099 features (99.9% sparse).
##          features
## docs        account auto bank call charg chase dai date dollar
##   3113204 1       1    2    2    1     1     1   3    1      1
##   3113208 0       1    0    6    3     5     0   0    1      1
##   3113804 0       0    0    0    0     0     0   2    2      0
##   3113805 0       1    0    0    0     0     0   0    0      0
##   3113807 0       2    0    0    0     1     0   0    0      0
##   3113808 0       0    0    0    0     0     0   0    0      0
## [ reached max_ndoc ... 117,208 more documents, reached max_nfeat ... 46,089 more features ]
```

The dataset of consumer complaints used in this book has been filtered to those submitted to the CFPB since 1 January 2019 that include a consumer complaint narrative (i.e., some submitted text).

Another way to represent our text data is to use [tf-idf](https://www.tidytextmining.com/tfidf.html) instead of word counts. This weighting for text features can often work better in predictive modeling.


```r
complaints %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  anti_join(get_stopwords()) %>%
  mutate(stem = wordStem(word)) %>%
  count(complaint_id, stem) %>%
  bind_tf_idf(stem, complaint_id, n) %>%
  cast_dfm(complaint_id, stem, tf_idf)
```

```
## Document-feature matrix of: 117,214 documents, 46,099 features (99.9% sparse).
##          features
## docs             account       auto       bank        call      charg
##   3113204 NA 0.008061739 0.09822024 0.04692567 0.015184546 0.02490318
##   3113208  0 0.001215924 0          0.02123290 0.006870693 0.01878029
##   3113804  0 0           0          0          0           0         
##   3113805  0 0.003113794 0          0          0           0         
##   3113807  0 0.034618057 0          0          0           0.05346859
##   3113808  0 0           0          0          0           0         
##          features
## docs           chase         dai        date      dollar
##   3113204 0.05115103 0.058265271 0.024506604 0.046154012
##   3113208 0          0           0.003696244 0.006961246
##   3113804 0          0.007001423 0.008834479 0          
##   3113805 0          0           0           0          
##   3113807 0          0           0           0          
##   3113808 0          0           0           0          
## [ reached max_ndoc ... 117,208 more documents, reached max_nfeat ... 46,089 more features ]
```

Notice that in either case, our final data structure is incredibly sparse and of high dimensionality with a huge number of features. Some modeling algorithms and the libraries which implement them can take advantage of the memory characteristics of sparse matrices for better performance; an example of this is regularized regression implemented in **glmnet**. Some modeling algorithms, including tree-based algorithms, do not perform better with sparse input, and then some libraries are not built to take advantage of sparse data structures, even if it would improve performance for those algorithms.

#### SPARSE VS. NON SPARSE MATRIX DIAGRAM GOES HERE

Linguists have long worked on vector models for language that can reduce the number of dimensions representing text data based on how people use language; the quote that opened this chapter dates to 1957. These kinds of dense word vectors are often called **word embeddings**.

## Understand word embeddings by finding them yourself

Word embeddings are a way to represent text data as numbers based on a huge corpus of text, capturing semantic meaning from words' context. 

\begin{rmdnote}
Modern word embeddings are based on a statistical approach to modeling
language, rather than a linguistics or rules-based approach.
\end{rmdnote}

We can determine these vectors for a corpus of text using word counts and matrix factorization, as outlined by @Moody2017. This approach is valuable because it allows practitioners to find word vectors for their own collections of text (with no need to rely on pre-trained vectors) using familiar techniques that are not difficult to understand. Let's walk through how to do this using tidy data principles and sparse matrices, on the dataset of CFPB complaints. First, let's filter out words that are used only rarely in this dataset and create a nested dataframe, with one row per complaint.


```r
tidy_complaints <- complaints %>%
  select(complaint_id, consumer_complaint_narrative) %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  add_count(word) %>%
  filter(n >= 50) %>%
  select(-n)

nested_words <- tidy_complaints %>%
  nest(words = c(word))

nested_words
```

```
## # A tibble: 117,170 x 2
##    complaint_id words             
##           <dbl> <list>            
##  1      3384392 <tibble [18 x 1]> 
##  2      3417821 <tibble [71 x 1]> 
##  3      3433198 <tibble [77 x 1]> 
##  4      3366475 <tibble [69 x 1]> 
##  5      3385399 <tibble [213 x 1]>
##  6      3444592 <tibble [19 x 1]> 
##  7      3379924 <tibble [121 x 1]>
##  8      3446975 <tibble [22 x 1]> 
##  9      3214857 <tibble [64 x 1]> 
## 10      3417374 <tibble [44 x 1]> 
## # ... with 117,160 more rows
```


Next, let’s create a `slide_windows()` function, using the `slide()` function from the **slider** package [@Vaughan2020] which implements fast sliding window computations written in C. Our new function identifies skipgram windows in order to calculate the skipgram probabilities, how often we find each word near each other word. We do this by defining a fixed-size moving window that centers around each word. Do we see `word1` and `word2` together within this window? We can calculate probabilities based on when we do or do not.

One of the arguments to this function is the `window_size`, which determines the size of the sliding window that moves through the text, counting up words that we find within the window. The best choice for this window size depends on your analytical question because it determines what kind of semantic meaning the embeddings capture. A smaller window size, like three or four, focuses on how the word is used and learns what other words are functionally similar. A larger window size, like ten, captures more information about the domain or topic of each word, not constrained by how functionally similar the words are [@Levy2014]. A smaller window size is also faster to compute.


```r
slide_windows <- function(tbl, window_size) {
  skipgrams <- slider::slide(
    tbl,
    ~.x,
    .after = window_size - 1,
    .step = 1,
    .complete = TRUE
  )

  safe_mutate <- safely(mutate)

  out <- map2(
    skipgrams,
    1:length(skipgrams),
    ~ safe_mutate(.x, window_id = .y)
  )

  out %>%
    transpose() %>%
    pluck("result") %>%
    compact() %>%
    bind_rows()
}
```

Now that we can find all the skipgram windows, we can calculate how often words occur on their own, and how often words occur together with other words. We do this using the point-wise mutual information (PMI), a measure of association that measures exactly what we described in the previous sentence; it's the logarithm of the probability of finding two words together, normalized for the probability of finding each of the words alone. We use PMI to measure which words occur together more often than expected based on how often they occurred on their own. 

For this example, let's use a window size of **four**.

\begin{rmdnote}
This next step is the computationally expensive part of finding word
embeddings with this method, and can take a while to run. Fortunately,
we can use the \textbf{furrr} package {[}@Vaughan2018{]} to take
advantage of parallel processing because identifying skipgram windows in
one document is independent from all the other documents.
\end{rmdnote}



```r
library(widyr)
library(furrr)

plan(multiprocess) ## for parallel processing

tidy_pmi <- nested_words %>%
  mutate(words = future_map(words, slide_windows, 4,
    .progress = TRUE
  )) %>%
  unnest(words) %>%
  unite(window_id, complaint_id, window_id) %>%
  pairwise_pmi(word, window_id)

tidy_pmi
```


```
## # A tibble: 4,818,402 x 3
##    item1   item2           pmi
##    <chr>   <chr>         <dbl>
##  1 systems transworld  7.09   
##  2 inc     transworld  5.96   
##  3 is      transworld -0.135  
##  4 trying  transworld -0.107  
##  5 to      transworld -0.00206
##  6 collect transworld  1.07   
##  7 a       transworld -0.516  
##  8 debt    transworld  0.919  
##  9 that    transworld -0.542  
## 10 not     transworld -1.17   
## # ... with 4,818,392 more rows
```

When PMI is high, the two words are associated with each other, likely to occur together. When PMI is low, the two words are not associated with each other, unlikely to occur together.


\begin{rmdtip}
The step above used \texttt{unite()}, a function from \textbf{tidyr}
that pastes multiple columns into one, to make a new column for
\texttt{window\_id} from the old \texttt{window\_id} plus the
\texttt{complaint\_id}. This new column tells us which combination of
window and complaint each word belongs to.
\end{rmdtip}

We can next determine the word vectors from the PMI values using singular value decomposition. Let's use the `widely_svd()` function in **widyr**, creating 100-dimensional word embeddings. This matrix factorization is much faster than the previous step of identifying the skipgram windows and calculating PMI.


```r
tidy_word_vectors <- tidy_pmi %>%
  widely_svd(
    item1, item2, pmi,
    nv = 100, maxit = 1000
  )

tidy_word_vectors
```

```
## # A tibble: 747,500 x 3
##    item1   dimension   value
##    <chr>       <int>   <dbl>
##  1 systems         1 0.0165 
##  2 inc             1 0.0191 
##  3 is              1 0.0202 
##  4 trying          1 0.0423 
##  5 to              1 0.00904
##  6 collect         1 0.0370 
##  7 a               1 0.0126 
##  8 debt            1 0.0430 
##  9 that            1 0.0136 
## 10 not             1 0.0213 
## # ... with 747,490 more rows
```

We have now successfully found word embeddings, with clear and understandable code. This is a real benefit of this approach; this approach is based on counting, dividing, and matrix decomposition and is thus easier to understand and implement than options based on deep learning. Training word vectors or embeddings, even with this straightforward method, still requires a large dataset (ideally, hundreds of thousands of documents or more) and a not insignificant investment of time and computational power. 

## Exploring CFPB word embeddings

Now that we have determined word embeddings for the dataset of CFPB complaints, let's explore them and talk about they are used in modeling. We have projected the sparse, high-dimensional set of word features into a more dense, 100-dimensional set of features. 

\begin{rmdnote}
Each word can be represented as a numeric vector in this new feature
space.
\end{rmdnote}

Which words are close to each other in this new feature space of word embeddings? Let's create a simple function that will find the nearest words to any given example in using our newly created word embeddings.


```r
nearest_neighbors <- function(df, token) {
  df %>%
    widely(~ . %*% (.[token, ]), sort = TRUE)(item1, dimension, value) %>%
    select(-item2)
}
```

This function takes the tidy word embeddings as input, along with a word (or token, more strictly) as a string. It uses matrix multiplication to find which words are closer or farther to the input word, and returns a dataframe sorted by similarity.

What words are closest to `"error"` in the dataset of CFPB complaints, as determined by our word embeddings?


```r
tidy_word_vectors %>%
  nearest_neighbors("error")
```

```
## # A tibble: 7,475 x 2
##    item1      value
##    <chr>      <dbl>
##  1 error     0.0373
##  2 issue     0.0236
##  3 problem   0.0235
##  4 issues    0.0194
##  5 errors    0.0187
##  6 mistake   0.0185
##  7 system    0.0170
##  8 problems  0.0151
##  9 late      0.0141
## 10 situation 0.0138
## # ... with 7,465 more rows
```

Errors, problems, issues, mistakes -- sounds bad!

What is closest to the word `"month"`?


```r
tidy_word_vectors %>%
  nearest_neighbors("month")
```

```
## # A tibble: 7,475 x 2
##    item1     value
##    <chr>     <dbl>
##  1 month    0.0597
##  2 payment  0.0408
##  3 months   0.0355
##  4 payments 0.0325
##  5 year     0.0314
##  6 days     0.0275
##  7 balance  0.0267
##  8 xx       0.0265
##  9 years    0.0262
## 10 monthly  0.0260
## # ... with 7,465 more rows
```

We see words about payments, along with other time periods such as days and years. Notice that we did not stem this text data (see Chapter \@ref(stemming)) but the word embeddings learned that singular and plural forms of words belong together.

What words are closest in this embedding space to `"fee"`?


```r
tidy_word_vectors %>%
  nearest_neighbors("fee")
```

```
## # A tibble: 7,475 x 2
##    item1      value
##    <chr>      <dbl>
##  1 fee       0.0762
##  2 fees      0.0605
##  3 charge    0.0421
##  4 interest  0.0410
##  5 charged   0.0387
##  6 late      0.0377
##  7 charges   0.0366
##  8 overdraft 0.0327
##  9 charging  0.0245
## 10 month     0.0220
## # ... with 7,465 more rows
```

We find words about interest, charges, and overdrafts.

Since we have found word embeddings via singular value decomposition, we can use these vectors to understand what principal components explain the most variation in the CFPB complaints.


```r
tidy_word_vectors %>%
  filter(dimension <= 24) %>%
  group_by(dimension) %>%
  top_n(12, abs(value)) %>%
  ungroup() %>%
  mutate(item1 = reorder_within(item1, value, dimension)) %>%
  ggplot(aes(item1, value, fill = as.factor(dimension))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~dimension, scales = "free_y", ncol = 4) +
  scale_x_reordered() +
  coord_flip() +
  labs(
    x = NULL, y = "Value",
    title = "First 24 principal components for text of CFPB complaints",
    subtitle = "Top words contributing to the components that explain the most variation"
  )
```

![(\#fig:embeddingpca)Word embeddings for Consumer Finance Protection Bureau complaints](word-embeddings_files/figure-latex/embeddingpca-1.pdf) 

It becomes very clear in Figure \@ref(fig:embeddingpca) that stop words have not been removed, but notice that we can learn meaningful relationships in how very common words are used. Component 12 shows us how common prepositions are often used with words like `"regarding"`, `"contacted"`, and `"called"`, while component 9 highlights the use of *different* common words when submitting a complaint about unethical, predatory, and/or deceptive practices. Stop words do carry information, and methods like determining word embeddings can make that information usable.

We created word embeddings and can explore them to understand our text dataset, but how do we use this vector representation in modeling? The classic and simplest approach is to treat each document as a collection of words and summarize the word embeddings into **document embeddings**, either using a mean or sum. Let's `count()` to find the sum here in our example.


```r
word_matrix <- tidy_complaints %>%
  count(complaint_id, word) %>%
  cast_sparse(complaint_id, word, n)

embedding_matrix <- tidy_word_vectors %>%
  cast_sparse(item1, dimension, value)

doc_matrix <- word_matrix %*% embedding_matrix

dim(doc_matrix)
```

```
## [1] 117170    100
```

We have a new matrix here that we can use as the input for modeling. Notice that we still have over 100,000 documents (we did lose a few complaints, compared to our example sparse matrices at the beginning of the chapter, when we filtered out rarely used words) but instead of tens of thousands of features, we have exactly 100 features.  

\begin{rmdnote}
These hundred features are the word embeddings we learned from the text
data itself.
\end{rmdnote}

If our word embeddings are of high quality, this translation of the high-dimensional space of words to the lower-dimensional space of the word embeddings allows our modeling based on such an input matrix to take advantage of the semantic meaning captured in the embeddings.

This is a straightforward method for finding and using word embeddings, based on counting and linear algebra. It is valuable both for understanding what word embeddings are and how they work, but also in many real-world applications. This is not the method to reach for if you want to publish an academic NLP paper, but is excellent for many applied purposes. Other methods for determining word embeddings include GloVe [@Pennington2014], implemented in R in the [text2vec](http://text2vec.org/) package [@Selivanov2018], word2vec [@Mikolov2013], and FastText [Bojanowski2016]. 

TODO maybe: mention https://github.com/mkearney/wactor???

## Use pre-trained word embeddings {#glove}

If your dataset is too small, you typically cannot train reliable word embeddings. 

\begin{rmdtip}
How small is too small? It is hard to make definitive statements because
being able to determind useful word embeddings depends on the semantic
and pragmatic details of \emph{how} words are used in any given dataset.
However, it may be unreasonable to expect good results with datasets
smaller than about a million words or tokens. (Here, we do not mean
about a million unique tokens, i.e.~the vocabulary size, but instead
about that many observations in the text data.)
\end{rmdtip}

In such situations, we can still use word embeddings for feature creation in modeling, just not embeddings that we determine ourselves from our own dataset. Instead, we can turn to **pre-trained** word embeddings, such as the GloVe word vectors trained on six billion tokens from Wikipedia and news sources. Several pre-trained GloVe vector representations are available in R via the [textdata](https://cran.r-project.org/package=textdata) package [@Hvitfeldt2020]. Let's use `dimensions = 100`, since we trained 100-dimensional word embeddings in the previous section.


```r
library(textdata)

glove6b <- embedding_glove6b(dimensions = 100)
glove6b
```


```
## # A tibble: 400,000 x 101
##    token      d1      d2      d3      d4      d5      d6      d7      d8      d9
##    <chr>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
##  1 "the" -0.0382 -0.245   0.728  -0.400   0.0832  0.0440 -0.391   0.334  -0.575 
##  2 ","   -0.108   0.111   0.598  -0.544   0.674   0.107   0.0389  0.355   0.0635
##  3 "."   -0.340   0.209   0.463  -0.648  -0.384   0.0380  0.171   0.160   0.466 
##  4 "of"  -0.153  -0.243   0.898   0.170   0.535   0.488  -0.588  -0.180  -1.36  
##  5 "to"  -0.190   0.0500  0.191  -0.0492 -0.0897  0.210  -0.550   0.0984 -0.201 
##  6 "and" -0.0720  0.231   0.0237 -0.506   0.339   0.196  -0.329   0.184  -0.181 
##  7 "in"   0.0857 -0.222   0.166   0.134   0.382   0.354   0.0129  0.225  -0.438 
##  8 "a"   -0.271   0.0440 -0.0203 -0.174   0.644   0.712   0.355   0.471  -0.296 
##  9 "\""  -0.305  -0.236   0.176  -0.729  -0.283  -0.256   0.266   0.0253 -0.0748
## 10 "'s"   0.589  -0.202   0.735  -0.683  -0.197  -0.180  -0.392   0.342  -0.606 
## # ... with 399,990 more rows, and 91 more variables: d10 <dbl>, d11 <dbl>,
## #   d12 <dbl>, d13 <dbl>, d14 <dbl>, d15 <dbl>, d16 <dbl>, d17 <dbl>,
## #   d18 <dbl>, d19 <dbl>, d20 <dbl>, d21 <dbl>, d22 <dbl>, d23 <dbl>,
## #   d24 <dbl>, d25 <dbl>, d26 <dbl>, d27 <dbl>, d28 <dbl>, d29 <dbl>,
## #   d30 <dbl>, d31 <dbl>, d32 <dbl>, d33 <dbl>, d34 <dbl>, d35 <dbl>,
## #   d36 <dbl>, d37 <dbl>, d38 <dbl>, d39 <dbl>, d40 <dbl>, d41 <dbl>,
## #   d42 <dbl>, d43 <dbl>, d44 <dbl>, d45 <dbl>, d46 <dbl>, d47 <dbl>,
## #   d48 <dbl>, d49 <dbl>, d50 <dbl>, d51 <dbl>, d52 <dbl>, d53 <dbl>,
## #   d54 <dbl>, d55 <dbl>, d56 <dbl>, d57 <dbl>, d58 <dbl>, d59 <dbl>,
## #   d60 <dbl>, d61 <dbl>, d62 <dbl>, d63 <dbl>, d64 <dbl>, d65 <dbl>,
## #   d66 <dbl>, d67 <dbl>, d68 <dbl>, d69 <dbl>, d70 <dbl>, d71 <dbl>,
## #   d72 <dbl>, d73 <dbl>, d74 <dbl>, d75 <dbl>, d76 <dbl>, d77 <dbl>,
## #   d78 <dbl>, d79 <dbl>, d80 <dbl>, d81 <dbl>, d82 <dbl>, d83 <dbl>,
## #   d84 <dbl>, d85 <dbl>, d86 <dbl>, d87 <dbl>, d88 <dbl>, d89 <dbl>,
## #   d90 <dbl>, d91 <dbl>, d92 <dbl>, d93 <dbl>, d94 <dbl>, d95 <dbl>,
## #   d96 <dbl>, d97 <dbl>, d98 <dbl>, d99 <dbl>, d100 <dbl>
```

We can transform these word embeddings into a more tidy format, using `pivot_longer()` from tidyr. Let's also give this tidied version the same column names as `tidy_word_vectors`, for convenience.


```r
tidy_glove <- glove6b %>%
  pivot_longer(contains("d"),
    names_to = "dimension"
  ) %>%
  rename(item1 = token)

tidy_glove
```

```
## # A tibble: 40,000,000 x 3
##    item1 dimension   value
##    <chr> <chr>       <dbl>
##  1 the   d1        -0.0382
##  2 the   d2        -0.245 
##  3 the   d3         0.728 
##  4 the   d4        -0.400 
##  5 the   d5         0.0832
##  6 the   d6         0.0440
##  7 the   d7        -0.391 
##  8 the   d8         0.334 
##  9 the   d9        -0.575 
## 10 the   d10        0.0875
## # ... with 39,999,990 more rows
```

We've already explored some sets of "synonyms" in the embedding space we determined ourselves from the CPFB complaints. What about this embedding space learned via the GloVe algorithm on a much larger dataset? We just need to make one change to our `nearest_neighbors()` function, because the matrices we are multiplying together are much larger this time.


```r
nearest_neighbors <- function(df, token) {
  df %>%
    widely(~ . %*% (.[token, ]), sort = TRUE, maximum_size = NULL)(item1, dimension, value) %>%
    select(-item2)
}
```

Pre-trained word embeddings are trained based on very large, general purpose English language datasets. Commonly used [word2vec embeddings](https://code.google.com/archive/p/word2vec/) are based on the Google News dataset, and commonly used [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (what we are using here) and [FastText embeddings](https://fasttext.cc/docs/en/english-vectors.html) are learned from the text of Wikipedia plus other sources. Keeping that in mind, what words are closest to `"error"` in the GloVe embeddings?


```r
tidy_glove %>%
  nearest_neighbors("error")
```

```
## # A tibble: 400,000 x 2
##    item1       value
##    <chr>       <dbl>
##  1 error        34.6
##  2 errors       28.1
##  3 data         19.8
##  4 inning       19.4
##  5 game         19.3
##  6 percentage   19.3
##  7 probability  19.2
##  8 unforced     19.1
##  9 fault        19.1
## 10 point        19.0
## # ... with 399,990 more rows
```

Instead of problems and mistakes like in the CFPB embeddings, we now see words related to sports, especially baseball, where an error is a certain kind of act recorded in statistics. This could present a challenge for using the GloVe embeddings with the CFPB text data.

What is closest to the word `"month"` in these pre-trained GloVe embeddings?


```r
tidy_glove %>%
  nearest_neighbors("month")
```

```
## # A tibble: 400,000 x 2
##    item1     value
##    <chr>     <dbl>
##  1 month      32.4
##  2 year       31.2
##  3 last       30.6
##  4 week       30.5
##  5 wednesday  29.6
##  6 tuesday    29.5
##  7 monday     29.3
##  8 thursday   29.1
##  9 percent    28.9
## 10 friday     28.9
## # ... with 399,990 more rows
```

Instead of words about payments, the GloVe results here focus on different time periods only.

What words are closest in the GloVe embedding space to `"fee"`?


```r
tidy_glove %>%
  nearest_neighbors("fee")
```

```
## # A tibble: 400,000 x 2
##    item1        value
##    <chr>        <dbl>
##  1 fee           39.8
##  2 fees          30.7
##  3 pay           26.6
##  4 $             26.4
##  5 salary        25.9
##  6 payment       25.9
##  7 £             25.4
##  8 tax           24.9
##  9 payments      23.8
## 10 subscription  23.1
## # ... with 399,990 more rows
```

The most similar words are, like with the CPFB embeddings, generally financial, but they are largely about salary and pay instead of about charges and overdrafts.

\begin{rmdwarning}
These examples highlight how pre-trained word embeddings can be useful
because of the incredibly rich semantic relationships they encode, but
also how these vector representations are often less than ideal for
specific tasks.
\end{rmdwarning}

If we do choose to use pre-trained word embeddings, how do we go about integrating them into a modeling workflow? Again, we can create simple document embeddings by treating each document as a collection of words and summarizing the word embeddings. The GloVe embeddings do not contain all the tokens in the CPFB complaints, and vice versa, so let's use `inner_join()` to match up our datasets.


```r
word_matrix <- tidy_complaints %>%
  inner_join(tidy_glove %>%
    distinct(item1) %>%
    rename(word = item1)) %>%
  count(complaint_id, word) %>%
  cast_sparse(complaint_id, word, n)

glove_matrix <- tidy_glove %>%
  inner_join(tidy_complaints %>%
    distinct(word) %>%
    rename(item1 = word)) %>%
  cast_sparse(item1, dimension, value)

doc_matrix <- word_matrix %*% glove_matrix

dim(doc_matrix)
```

```
## [1] 117163    100
```

Since these GloVe embeddings had the same number of dimensions as the word embeddings we found ourselves (100), we end up with the same number of columns as before but with slightly fewer documents in the dataset. We have lost documents which contain only words not included in the GloVe embeddings.

## Fairness and word embeddings {#fairnessembeddings}

Perhaps more than any of the other preprocessing steps this book has covered so far, using word embeddings opens an analysis or model up to the possibility of being influenced by systemic unfairness and bias. 

\begin{rmdwarning}
Embeddings are trained or learned from a large corpus of text data, and
whatever human prejudice or bias exists in the corpus becomes imprinted
into the vector data of the embeddings.
\end{rmdwarning}

This is true of all machine learning to some extent (models learn, reproduce, and often amplify whatever biases exist in training data) but this is literally, concretely true of word embeddings. @Caliskan2016 show how the GloVe word embeddings (the same embeddings we used in Section \@ref(glove)) replicate human-like semantic biases.

- African American first names are associated with more unpleasant feelings than European American first names.
- Women's first names are more associated with family and men's first names are more associated with career.
- Terms associated with women are more associated with the arts and terms associated with men are more associated with science.

Results like these have been confirmed over and over again, such as when @Bolukbasi2016 demonstrated gender stereotypes in how word embeddings encode professions or when Google Translate [exhibited apparently sexist behavior when translating text from languages with no gendered pronouns](https://twitter.com/seyyedreza/status/935291317252493312). ^[Google has since [worked to correct this problem.](https://www.blog.google/products/translate/reducing-gender-bias-google-translate/)] @Garg2018 even used the way bias and stereotypes can be found in word embeddings to quantify how social attitudes towards women and minorities have changed over time. 

Remember that word embeddings are *learned* or trained from some large dataset of text; this training data is the source of the biases we observe when applying word embeddings to NLP tasks. One common dataset used to train large embedding models is the text of [Wikipedia](https://en.wikipedia.org/wiki/Gender_bias_on_Wikipedia), but Wikipedia itself has problems with, for example, gender bias. Some of the gender discrepancies on Wikipedia can be attributed to social and historical factors, but some can be attributed to the site mechanics of Wikipedia itself [@Wagner2016].

\begin{rmdtip}
It's safe to assume that any large corpus of language will contain
latent structure reflecting the biases of the people who generated that
language.
\end{rmdtip}

When embeddings with these kinds of stereotypes are used as a preprocessing step in training a predictive model, the final model can exhibit racist, sexist, or otherwise biased characteristics. @Speer2017 demonstrated how using pre-trained word embeddings to train a straightforward sentiment analysis model can result in text such as 

> "Let's go get Italian food"

being scored much more positively than text such as

> "Let's go get Mexican food"

because of characteristics of the text the word embeddings were trained on.

## Using word embeddings in the real world

Given these profound and fundamental challenges with word embeddings, what options are out there? First, consider not using word embeddings when building a text model. Depending on the particular analytical question you are trying to answer, another numerical representation of text data (such as word frequencies or tf-idf of single words or n-grams) may be more appropriate. Consider this option even more seriously if the model you want to train is already entangled with issues of bias, such as the sentiment analysis example in Section \@ref(fairnessembeddings).

Consider whether finding your own word embeddings, instead of relying on pre-trained embeddings created using an algorithm such as GloVe or word2vec, may help you. Building your own vectors is likely to be a good option when the text domain you are working in is **specific** rather than general purpose; some examples of such domains could include customer feedback for a clothing e-commerce site, comments posted on a coding Q&A site, or legal documents. 

Learning good quality word embeddings is only realistic when you have a large corpus of text data (say, a million tokens) but if you have that much data, it is possible that embeddings learned from scratch based on your own data may not exhibit the same kind of semantic biases that exist in pre-trained word embeddings. Almost certainly there will be some kind of bias latent in any large text corpus, but when you use your own training data for learning word embeddings, you avoid the problem of *adding* historic, systemic prejudice from general purpose language datasets.

\begin{rmdnote}
You can use the same approaches discussed in this chapter to check any
new embeddings for dangerous biases such as racism or sexism.
\end{rmdnote}

NLP researchers have also proposed methods for debiasing embeddings. @Bolukbasi2016 aim to remove stereotypes by postprocessing pre-trained word vectors, choosing specific sets of words that are reprojected in the vector space so that some specific bias, such as gender bias, is mitigated. This is the most established method for reducing bias in embeddings to date, although other methods have been proposed as well, such as augmenting data with counterfactuals [@Lu2018]. Recent work [@Ethayarajh2019] has explored whether the association tests used to measure bias are even useful, and under what conditions debiasing can be effective.

Other researchers, such as @Caliskan2016, suggest that corrections for fairness should happen at the point of **decision** or action rather than earlier in the process of modeling, such as preprocessing steps like building word embeddings. The concern is that methods for debiasing word embeddings may allow the stereotypes to seep back in, and more recent work shows that this is exactly what can happen. @Gonen2019 highlight how pervasive and consistent gender bias is across different word embedding models, *even after* applying current debiasing methods.

## Summary

Mapping words (or other tokens) to an embedding in a special vector space is a powerful approach in natural language processing. This chapter started from fundamentals to demonstrate how to determine word embeddings from a text dataset, but a whole host of highly sophisticated techniques have been built on this foundation. For example, document embeddings can be learned from text directly [@Le2014] rather than summarized from word embeddings. More recently, embeddings have acted as one part of language models with transformers like ULMFiT [@Howard2018] and ELMo [@Peters2018]. It's important to keep in mind that even more advanced natural language algorithms, such as these language models with transformers, also exhibit such systemic biases [@Sheng2019].

### In this chapter, you learned:

- what a word embedding is and why we use them
- how to determine word embeddings from a text dataset
- how the vector space of word embeddings encodes word similarity
- about a simple strategy to find document similarity
- how to handle pre-trained word embeddings
- why word embeddings carry historic and systemic bias
- about approaches for debiasing word embeddings
