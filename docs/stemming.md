# Stemming {#stemming}



When we deal with text, often documents contain different versions of one base word, often called a **stem**. "The Fir-Tree", for example, contains more than one version (i.e., inflected form) of the word `"tree"`.


```r
library(hcandersenr)
library(tidyverse)
library(tidytext)

fir_tree <- hca_fairytales() %>%
  filter(
    book == "The fir tree",
    language == "English"
  )

tidy_fir_tree <- fir_tree %>%
  unnest_tokens(word, text) %>%
  anti_join(get_stopwords())

tidy_fir_tree %>%
  count(word, sort = TRUE) %>%
  filter(str_detect(word, "^tree"))
```

```
## # A tibble: 3 x 2
##   word       n
##   <chr>  <int>
## 1 tree      76
## 2 trees     12
## 3 tree's     1
```

Trees, we see once again, are important in this story; the singular form appears 76 times and the plural form appears twelve times. (We'll come back to how we might handle the apostrophe in `"tree's"` later in this chapter.)

What if we aren't interested in the difference between `"trees"` and `"tree"` and we want to treat both together? That idea is at the heart of **stemming**, the process of identifying the base word (or stem) for a dataset of words. Stemming is concerned with the linguistics subfield of morphology, how words are formed. In this example, `"trees"` would lose its letter `"s"` while `"tree"` stays the same. If we counted word frequencies again after stemming, we would find that there are 88 occurrences of the stem `"tree"` (89, if we also find the stem for `"tree's"`).

## How to stem text in R

There have been many algorithms built for stemming words over the past half century or so; we'll focus on two approaches. The first is the stemming algorithm of @Porter80, probably the most widely used stemmer for English. Porter himself released the algorithm implemented in the framework [Snowball](https://snowballstem.org/) with an open-source license; you can use it from R via the [SnowballC](https://cran.r-project.org/package=SnowballC) package. (It has been extended to languages other than English as well.)


```r
library(SnowballC)

tidy_fir_tree %>%
  mutate(stem = wordStem(word)) %>%
  count(stem, sort = TRUE)
```

```
## # A tibble: 570 x 2
##    stem        n
##    <chr>   <int>
##  1 tree       88
##  2 fir        34
##  3 littl      23
##  4 said       22
##  5 stori      16
##  6 thought    16
##  7 branch     15
##  8 on         15
##  9 came       14
## 10 know       14
## # ... with 560 more rows
```

Take a look at those stems. Notice that we do now have 88 incidences of "tree". Also notice that some words don't look like they are spelled as real words; this is normal and expected with this stemming algorithm. The Porter algorithm identifies the stem of both "story" and "stories" as "stori", not a regular English word but instead a special stem object.

\BeginKnitrBlock{rmdtip}<div class="rmdtip">If you want to tokenize *and* stem your text data, you can try out the function `tokenize_word_stems()` from the tokenizers package, which implements Porter stemming just like what we demonstrated here. For more on tokenization, see Chapter \@ref(tokenization).
</div>\EndKnitrBlock{rmdtip}

The Porter stemmer is an algorithm that starts with a word and ends up with a single stem, but that's not the only kind of stemmer out there. Another class of stemmer are dictionary-based stemmers. One such stemmer is the stemming algorithm of the [Hunspell](http://hunspell.github.io/) library. The "Hun" in Hunspell stands for Hungarian; this set of NLP algorithms was originally written to handle Hungarian but has since been extended to handle many languages with compound words and complicated morphology. The Hunspell library is used mostly as a spell checker, but as part of identifying correct spellings, this library identifies word stems as well. You can use the Hunspell library from R via the [hunspell](https://cran.r-project.org/package=hunspell) package.


```r
library(hunspell)

tidy_fir_tree %>%
  mutate(stem = hunspell_stem(word)) %>%
  unnest(stem) %>%
  count(stem, sort = TRUE)
```

```
## # A tibble: 595 x 2
##    stem       n
##    <chr>  <int>
##  1 tree      89
##  2 fir       34
##  3 little    23
##  4 said      22
##  5 story     16
##  6 branch    15
##  7 one       15
##  8 came      14
##  9 know      14
## 10 now       14
## # ... with 585 more rows
```

Notice that the code here is a little different (we had to use `unnest()`) and that the results are a little different. We have only real English words, and we have more total rows in the result. What happened?


```r
hunspell_stem("discontented")
```

```
## [[1]]
## [1] "contented" "content"
```

We have **two** stems! This stemmer works differently; it uses both morphological analysis of a word and existing dictionaries to find possible stems. It's possible to end up with more than one, and it's possible for a stem to be a word that is not related by meaning to the original word. For example, one of the stems of "number" is "numb" with this library. The Hunspell library was built to be a spell checker, so depending on your analytical purposes, it may not be an appropriate choice.

## Should you use stemming at all?

You will often see stemming as part of NLP pipelines, sometimes without much comment about when it is helpful or not. We encourage you to think of stemming as a preprocessing step in text modeling, one that must be thought through and chosen (or not) with good judgment.

Why does stemming often help, if you are training a machine learning model for text? Stemming **reduces the sparsity** of text data. Let's see this in action, with a dataset of United States Supreme Court opinions available in the [**scotus**](https://github.com/EmilHvitfeldt/scotus) package. How many words are there, after removing a standard dataset of stopwords?


```r
library(scotus)

tidy_scotus <- scotus_sample %>%
  unnest_tokens(word, text) %>%
  anti_join(get_stopwords())

tidy_scotus %>%
  count(word, sort = TRUE)
```

```
## # A tibble: 89,617 x 2
##    word       n
##    <chr>  <int>
##  1 court  83428
##  2 v      59193
##  3 state  45415
##  4 states 39119
##  5 case   35319
##  6 act    32506
##  7 s.ct   32003
##  8 u.s    31376
##  9 united 30803
## 10 upon   30533
## # ... with 89,607 more rows
```

There are 89,617 distinct words in this dataset we have created (after removing stopwords) but notice that even in the most common words we see a pair like `"state"` and `"states"`. A common data structure for modeling, and a helpful mental model for thinking about the sparsity of text data, is a matrix. Let's `cast()` this tidy data to a sparse matrix (technically, a document-feature matrix object from the [quanteda](https://cran.r-project.org/package=quanteda) package).


```r
tidy_scotus %>%
  count(case_name, word) %>%
  cast_dfm(case_name, word, n)
```

```
## Document-feature matrix of: 6,060 documents, 89,617 features (99.6% sparse).
##                                                                                   features
## docs                                                                               02
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 1
##                                                                                   features
## docs                                                                               284
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               3d
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 2
##                                                                                   features
## docs                                                                               71
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 0
##                                                                                   features
## docs                                                                               7th
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               807
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               appeals
##   1-800-flowers.com, Inc. v. Jahn                                                        1
##   10 East 40th Street Building, Inc. v. Callus                                           1
##   149 Madison Ave. Corp. v. Asselta                                                      2
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia                   0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States       3
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                      0
##                                                                                   features
## docs                                                                               c
##   1-800-flowers.com, Inc. v. Jahn                                                  1
##   10 East 40th Street Building, Inc. v. Callus                                     0
##   149 Madison Ave. Corp. v. Asselta                                                1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia             0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States 1
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                1
##                                                                                   features
## docs                                                                               certiorari
##   1-800-flowers.com, Inc. v. Jahn                                                           2
##   10 East 40th Street Building, Inc. v. Callus                                              0
##   149 Madison Ave. Corp. v. Asselta                                                         1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia                      1
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States          3
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                         1
##                                                                                   features
## docs                                                                               cir
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       4
##   149 Madison Ave. Corp. v. Asselta                                                  1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  1
## [ reached max_ndoc ... 6,054 more documents, reached max_nfeat ... 89,607 more features ]
```

Look at the sparsity of this matrix. It's high! Think of this sparsity as the sparsity of data that we will want to use to build a supervised machine learning model.

What if instead we use stemming as a preprocessing step here?


```r
tidy_scotus %>%
  mutate(stem = wordStem(word)) %>%
  count(case_name, stem) %>%
  cast_dfm(case_name, stem, n)
```

```
## Document-feature matrix of: 6,060 documents, 69,619 features (99.5% sparse).
##                                                                                   features
## docs                                                                               02
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 1
##                                                                                   features
## docs                                                                               284
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               3d
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 2
##                                                                                   features
## docs                                                                               71
##   1-800-flowers.com, Inc. v. Jahn                                                   1
##   10 East 40th Street Building, Inc. v. Callus                                      0
##   149 Madison Ave. Corp. v. Asselta                                                 0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia              0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States  0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                 0
##                                                                                   features
## docs                                                                               7th
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               807
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       0
##   149 Madison Ave. Corp. v. Asselta                                                  0
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  0
##                                                                                   features
## docs                                                                               appeal
##   1-800-flowers.com, Inc. v. Jahn                                                       1
##   10 East 40th Street Building, Inc. v. Callus                                          1
##   149 Madison Ave. Corp. v. Asselta                                                     2
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia                  0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States      3
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                     0
##                                                                                   features
## docs                                                                               c
##   1-800-flowers.com, Inc. v. Jahn                                                  1
##   10 East 40th Street Building, Inc. v. Callus                                     0
##   149 Madison Ave. Corp. v. Asselta                                                1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia             0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States 1
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                1
##                                                                                   features
## docs                                                                               certiorari
##   1-800-flowers.com, Inc. v. Jahn                                                           2
##   10 East 40th Street Building, Inc. v. Callus                                              0
##   149 Madison Ave. Corp. v. Asselta                                                         1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia                      1
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States          3
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                         1
##                                                                                   features
## docs                                                                               cir
##   1-800-flowers.com, Inc. v. Jahn                                                    1
##   10 East 40th Street Building, Inc. v. Callus                                       4
##   149 Madison Ave. Corp. v. Asselta                                                  1
##   1690 Cobb L. L. C., Dba Waterpipe World v. City of Marietta, Georgia               0
##   2,606.84 Acres of Land in Tarrant County, Texas, and Frank Corn v. United States   0
##   3m Co., Fka Minnesota Mining & Manufacturing Co. v. Lepage's Inc.                  1
## [ reached max_ndoc ... 6,054 more documents, reached max_nfeat ... 69,609 more features ]
```

We reduced the sparsity of our data by about 1%, and the number of word features by many thousands. Why is it possibly helpful to make our data more dense? Common sense says that reducing the number of word features in our dataset so dramatically will improve the performance of any machine learning model we train with it, *assuming that we haven't lost any important information by stemming*.

There is a growing body of academic research demonstrating that stemming can be counterproductive for text modeling. For example, @Schofield16 and related work explore how choices around stemming and other preprocessing steps don't help and can actually hurt performance when training topic models for text. From @Schofield16 specifically,

> Despite their frequent use in topic modeling, we find that stemmers produce no meaningful improvement in likelihood and coherence and in fact can degrade topic stability.

Topic modeling is an example of unsupervised machine learning for text and is not the same as the predictive modeling approaches we'll be focusing on in this book, but the lesson remains that stemming may or may not be beneficial for any specific context. As we work through the rest of this chapter and learn more about stemming, consider what information we lose when we stem text in exchange for reducing the number of word features. Stemming can be helpful in some contexts, but typical stemming algorithms are somewhat aggressive and have been built to favor sensitivity (or recall, or the true positive rate) at the expense of specificity (or precision, or the true negative rate). Most common stemming algorithms you are likely to encounter will successfully reduce words to stems (i.e., not leave extraneous word endings on the words) but at the expense of collapsing some words with dramatic differences in meaning, semantics, use, etc. to the same stems. Examples of the latter are numerous, but some include:

- meaning and mean
- likely, like, liking
- university and universe

## Understand a stemming algorithm

If stemming is going to be in our NLP toolbox, it's worth sitting down with one approach in detail to understand how it works under the hood. The Porter stemming algorithm is so approachable that we can walk through its outline in less than a page or so. It involves five steps, and the idea of a word **measure**.


Think of any word as made up alternating groups of vowels $V$ and consonants $C$. One or more vowels together are one instance of $V$, and one or more consonants togther are one instance of $C$. We can write any word as

$$[C](VC)^m[V]$$
where $m$ is called the "measure" of the word. The first $C$ and the last $V$ in brackets are optional. In this framework, we could write out the word `"tree"` as

$$CV$$

with $C$ being "tr" and $V$ being "ee"; it's an `m = 0` word. We would write out the word `"algorithms"` as 

$$VCVCVC$$
and it is an `m = 3` word.

- The first step of the Porter stemmer is (perhaps this seems like cheating) actually made of three substeps working with plural and past participle word endings. In the first substep (1a), "sses" is replaced with "ss", "ies" is replaced with "i", and final single "s" letters are removed. The second substep (1b) depends on the measure of the word `m` but works with endings like "eed", "ed", "ing", adding "e" back on to make endings like "ate", "ble", and "ize" when appropriate. The third substep (1c) replaces "y" with "i" for words of a certain `m`.
- The second step of the Porter stemmer takes the output of the first step and regularizes a set of 20 endings. In this step, "ization" goes to "ize", "alism" goes to "al", "aliti" goes to "al" (notice that the ending "i" there came from the first step), and so on for the other 17 endings.
- The third step again processes the output, using a list of seven endings. Here, "ical" and "iciti" both go to "ic", "ful" and "ness" are both removed, and so forth for the three other endings in this step.
- The fourth step involves a longer list of endings to deal with again (19), and they are all removed. Endings like "ent", "ism", "ment", and more are removed in this step.
- The fifth and final step has two substeps, both which depend on the measure `m` of the word. In this step, depending on `m`, final "e" letters are sometimes removed and final double letters are sometimes removed.


\begin{rmdnote}
How would this work for a few example words? The word ``supervised''
loses its ``ed'' in step 1b and is not touched by the rest of the
algorithm, ending at ``supervis''. The word ``relational'' changes
``ational'' to ``ate'' in step 2 and loses its final ``e'' in step 5,
ending at ``relat''. Notice that neither of these results are regular
English words, but instead special stem objects. This is expected.
\end{rmdnote}

This algorithm was first published in @Porter80 and is still broadly used; read @Willett06 for background on how and why it has become a stemming standard. We can reach even *further* back and examine what is considered the first ever published stemming algorithm in @Lovins68. The domain Lovins worked in was engineering, so her approach was particularly suited to technical terms. This algorithm uses much larger lists of word endings, conditions, and rules than the Porter algorithm and, although considered old-fashioned, is actually faster!

\begin{rmdwarning}
Does stemming only work for English? Far from it! Check out the
\href{https://snowballstem.org/algorithms/german/stemmer.html}{steps of
a Snowball stemming algorithm for German}.
\end{rmdwarning}


## Handling punctuation when stemming

Punctuation contains information that can be used in text analysis. Punctuation *is* typically less information-dense than the words themselves and thus it is often removed early in a text mining analysis project, but it's worth thinking through the impact of punctuation specifically on stemming. Think about words like `"they're"` and `"child's"`.

We've already seen how punctuation and stemming can interact with our small example of "The Fir-Tree"; none of the stemming strategies we've discussed so far have recognized `"tree's"` as belonging to the same stem as `"trees"` and `"tree"`.


```r
tidy_fir_tree %>%
  count(word, sort = TRUE) %>%
  filter(str_detect(word, "^tree"))
```

```
## # A tibble: 3 x 2
##   word       n
##   <chr>  <int>
## 1 tree      76
## 2 trees     12
## 3 tree's     1
```

It is possible to split tokens not only on white space but **also** on punctuation, using a regular expression (see Appendix \@ref(regexp)). 


```r
fir_tree_counts <- fir_tree %>%
  unnest_tokens(word, text, token = "regex", pattern = "\\s+|[[:punct:]]+") %>%
  anti_join(get_stopwords()) %>%
  mutate(stem = wordStem(word)) %>%
  count(stem, sort = TRUE)

fir_tree_counts
```

```
## # A tibble: 572 x 2
##    stem        n
##    <chr>   <int>
##  1 tree       89
##  2 fir        34
##  3 littl      23
##  4 said       22
##  5 stori      16
##  6 thought    16
##  7 branch     15
##  8 on         15
##  9 came       14
## 10 know       14
## # ... with 562 more rows
```

Now we are able to put all these related words together, having identified them with the same stem.


```r
fir_tree_counts %>%
  filter(str_detect(stem, "^tree"))
```

```
## # A tibble: 1 x 2
##   stem      n
##   <chr> <int>
## 1 tree     89
```

Handling punctuation in this way further reduces sparsity in word features. Whether this kind of tokenization and stemming strategy is a good choice in any particular data analysis situation depends on the particulars of the text characteristics.

## Compare some stemming options

Let's compare a few simple stemming algorithms and see what results we end with. Let's look at "The Fir-Tree", specifically the tidied dataset from which we have removed stop words. Let's compare three very straightforward stemming approaches.

- **Only remove final instances of the letter "s".** This probably strikes you as not a great idea after our discussion in this chapter, but it is something that people try in real life, so let's see what the impact is.
- **Handle plural endings with slightly more complex rules.** These rules are the same as step 1a of Porter stemming.
- **Implement actual Porter stemming.** We can now compare to the most commonly used stemming algorithm in English.


```r
stemming <- tidy_fir_tree %>%
  select(-book, -language) %>%
  mutate(
    `Remove S` = str_remove(word, "s$"),
    `Plural endings` = case_when(
      str_detect(word, "sses$") ~
      str_replace(word, "sses$", "ss"),
      str_detect(word, "ies$") ~
      str_replace(word, "ies$", "y"),
      str_detect(word, "ss$") ~
      word,
      str_detect(word, "s$") ~
      str_remove(word, "s$"),
      TRUE ~ word
    ),
    `Porter stemming` = wordStem(word)
  ) %>%
  rename(`Original word` = word)

stemming
```

```
## # A tibble: 1,547 x 4
##    `Original word` `Remove S` `Plural endings` `Porter stemming`
##    <chr>           <chr>      <chr>            <chr>            
##  1 far             far        far              far              
##  2 forest          forest     forest           forest           
##  3 warm            warm       warm             warm             
##  4 sun             sun        sun              sun              
##  5 fresh           fresh      fresh            fresh            
##  6 air             air        air              air              
##  7 made            made       made             made             
##  8 sweet           sweet      sweet            sweet            
##  9 resting         resting    resting          rest             
## 10 place           place      place            place            
## # ... with 1,537 more rows
```

Figure \@ref(fig:stemmingresults) shows the results of these stemming strategies. All successfully handled the transition from `"trees"` to `"tree"` in the same way, but we have different results for `"stories"` to `"story"` or `"stori"` or `"storie"`, different handling of `"branches"`, and more. There are subtle differences in the output of even these straightforward stemming approaches that can effect the transformation of text features for modeling.


```r
stemming %>%
  gather(Type, Result, `Remove S`:`Porter stemming`) %>%
  mutate(Type = fct_inorder(Type)) %>%
  count(Type, Result) %>%
  group_by(Type) %>%
  top_n(20, n) %>%
  ungroup() %>%
  ggplot(aes(fct_reorder(Result, n),
    n,
    fill = Type
  )) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~Type, scales = "free_y") +
  coord_flip() +
  labs(x = NULL, y = "Frequency")
```

![(\#fig:stemmingresults)Results for three different stemming strategies](stemming_files/figure-latex/stemmingresults-1.pdf) 

Porter stemming is the most different from the other two approaches. In the top twenty words here, we don't see much difference between removing only the letter "s" and taking a slightly more sophisticated approach to plural endings. In what situations *do* we see a difference?


```r
stemming %>%
  filter(`Remove S` != `Plural endings`) %>%
  distinct(`Remove S`, `Plural endings`, .keep_all = TRUE)
```

```
## # A tibble: 10 x 4
##    `Original word` `Remove S`  `Plural endings` `Porter stemming`
##    <chr>           <chr>       <chr>            <chr>            
##  1 raspberries     raspberrie  raspberry        raspberri        
##  2 strawberries    strawberrie strawberry       strawberri       
##  3 less            les         less             less             
##  4 brightness      brightnes   brightness       bright           
##  5 faintness       faintnes    faintness        faint            
##  6 happiness       happines    happiness        happi            
##  7 ladies          ladie       lady             ladi             
##  8 babies          babie       baby             babi             
##  9 princess        princes     princess         princess         
## 10 stories         storie      story            stori
```

We also see situations where the same sets of original words are bucketed differently (not just with different stem labels) under different stemming strategies. In the following very small example, two of the strategies bucket these words into two stems while one strategy buckets them into one stem.


```r
stemming %>%
  gather(Type, Result, `Remove S`:`Porter stemming`) %>%
  filter(Result %in% c("come", "coming")) %>%
  distinct(`Original word`, Type, Result)
```

```
## # A tibble: 9 x 3
##   `Original word` Type            Result
##   <chr>           <chr>           <chr> 
## 1 come            Remove S        come  
## 2 comes           Remove S        come  
## 3 coming          Remove S        coming
## 4 come            Plural endings  come  
## 5 comes           Plural endings  come  
## 6 coming          Plural endings  coming
## 7 come            Porter stemming come  
## 8 comes           Porter stemming come  
## 9 coming          Porter stemming come
```

These different characteristics can either be positive or negative, depending on the nature of the text being modeled and the analytical question being pursued.

\begin{rmdwarning}
Language use is connected to culture and identity. How might the results
of stemming strategies be different for text created with the same
language (like English) but in different social or cultural contexts, or
by people with different identities? With what kind of text do you think
stemming algorithms behave most consistently, or most as expected? What
impact might that have on text modeling?
\end{rmdwarning}


## Lemmatization and stemming

When people use the word "stemming" in natural language processing, they typically mean a system like the one we've been describing in this chapter, with rules, conditions, heuristics, and lists of word endings. Think of stemming as typically implemented in NLP as **rule-based**, operating on the word by itself. There is another option for normalizing words to a root that takes a different approach. Instead of using rules to cut words down to their stems, lemmatization uses knowledge about a language's structure to reduce words down to their lemmas, the canonical or dictionary forms of words. Think of lemmatization as typically implemented in NLP as **linguistics-based**, operating on the word in its context.

Lemmatization requires more information than the rule-based stemmers we've discussed so far. We need to know what part of speech a word is to correctly identify its lemma ^[Part-of-speech information is also sometimes used directly in machine learning], and we also need more information about what words mean in their contexts. Often lemmatizers use a rich lexical database like [WordNet](https://wordnet.princeton.edu/) as a way to look up word meanings for a given part-of-speech use [@Miller95]. Notice that lemmatization involves more linguistic knowledge of a language than stemming. 

\begin{rmdtip}
How does lemmatization work in languages other than English? Lookup
dictionaries connecting words, lemmas, and parts of speech for languages
other than English have been developed as well.
\end{rmdtip}

A modern, efficient implementation for lemmatization is available in the excellent [spaCy](https://spacy.io/) library [@spacy2], which is written in Python. NLP practitioners who work with R can use this library via the [spacyr](http://spacyr.quanteda.io/) package [@Benoit19], the [cleanNLP](https://statsmaths.github.io/cleanNLP/) package [@Arnold17], or as an "engine" in the [textrecipes](https://tidymodels.github.io/textrecipes/dev/) package [@textrecipes]. Chapter TODO demonstrates how to use textrecipes with spaCy as an engine and include lemmas as features for modeling. You might also consider using spaCy directly in R Markdown [via its Python engine](https://rstudio.github.io/reticulate/articles/r_markdown.html). 

Implementing lemmatization is slower and more complex than stemming. Just like with stemming, lemmatization often improves the true positive rate (or recall) but at the expense of the true negative rate (or precision) compared to not using lemmatization, but typically less so than stemming.

## Stemming and stop words

Our deep dive into stemming came *after* our chapters on tokenization (Chapter \@ref(tokenization)) and stop words (Chapter \@ref(stopwords)) because this is typically when you will want to implement stemming, if appropriate to your analytical question. Stop word lists are usually unstemmed, so you need to remove stop words before stemming text data. For example, the Porter stemming algorithm transforms words like `"themselves"` to `"themselv"`, so stemming first would leave you without the ability to match up to the commonly used stop word lexicons.

A handy trick is to use the following function on your stop word list to return the words that don't have a stemmed version in the list. If the function returns a length 0 vector then you can stem and remove stop words in any order.


```r
library(stopwords)
not_stemmed_in <- function(x) {
  x[!SnowballC::wordStem(x) %in% x]
}

not_stemmed_in(stopwords(source = "snowball"))
```

```
##  [1] "ourselves"  "yourselves" "his"        "they"       "themselves"
##  [6] "this"       "are"        "was"        "has"        "does"      
## [11] "you're"     "he's"       "she's"      "it's"       "we're"     
## [16] "they're"    "i've"       "you've"     "we've"      "they've"   
## [21] "let's"      "that's"     "who's"      "what's"     "here's"    
## [26] "there's"    "when's"     "where's"    "why's"      "how's"     
## [31] "because"    "during"     "before"     "above"      "once"      
## [36] "any"        "only"       "very"
```

Here we see that many of the words that are lost are the contractions.

## Summary

In this chapter, we explored stemming, the practice of identifying and extracting the base or stem for a word using rules and heuristics. Stemming reduces the sparsity of text data which can be helpful when training models, but at the cost of throwing information away. Lemmatization is another way to normalize words to a root, based on language structure and how words are used in their context.

### In this chapter, you learned:

- about the most broadly used stemming algorithms
- how to implement stemming
- that stemming changes the sparsity or feature space of text data
- the differences between stemming and lemmatization

