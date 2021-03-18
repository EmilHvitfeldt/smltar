# Stop words {#stopwords}



Once we have tokenized text into words, it often becomes clear that not all of these words carry the same amount of information with them, if any information at all. Words that carry little (or perhaps no) meaningful information are called **stop words**. It is common advice and practice to remove stop words for various NLP tasks, but the task of stop word removal is more nuanced than many resources may lead you to believe. In this chapter, we will investigate what a stop word list is, the differences between them, and the effects of using them in your preprocessing workflow.

The concept of stop words has a long history with Hans Peter Luhn credited with coining the term in 1960 [@Luhn1960]. Examples of these words in English are "a", "the", "of", and "didn't". These words are very common and typically don't add much to the meaning of a text but instead ensure the structure of a sentence is sound. 

\begin{rmdtip}
Thinking of words as being either informative or non-informative is
quite limiting, and we prefer to consider words as having a more fluid
or continuous amount of information associated with them. This
informativeness is context specific as well.
\end{rmdtip}

Historically, one of the main reasons for removing stop words was to decrease computational time for text mining; it can be regarded as a dimensionality reduction of text data and was commonly used in search engines to give better results [@Huston2010].

## Using premade stop word lists

A quick solution to getting a list of stop words is to use one that is already created for you. This is appealing because it requires a low level of effort, but be aware that not all lists are created equal. @nothman-etal-2018-stop found some alarming results in a study of 52 stop word lists available in open source software packages. Their unexpected findings included how different stop word lists have a varying number of words depending on the specificity of the list. Among some of the more grave issues were misspellings ("fify" instead of "fifty"), the inclusion of clearly informative words such as "computer" and "cry", and internal inconsistencies such as including the word "has" but not the word "does". This is not to say that you should never use a stop word list that has been included in an open source software project. However, you should always inspect and verify the list you are using, both to make sure it hasn't changed since you used it last, and also to check that it is appropriate for your use case.

There is a broad selection of stop word lists available today. For the purpose of this chapter we will focus on three lists of English stop words provided by the **stopwords** package [@R-stopwords]. The first is from the SMART (System for the Mechanical Analysis and Retrieval of Text) Information Retrieval System, an information retrieval system developed at Cornell University in the 1960s [@Lewis2014]. The second is the English Snowball stop word list [@porter2001snowball], and the last is the English list from the [Stopwords ISO](https://github.com/stopwords-iso/stopwords-iso) collection. These stop word lists are all considered general purpose and not domain specific.

Before we start delving into the content inside the lists, let's take a look at how many words are included in each.


```r
library(stopwords)
length(stopwords(source = "smart"))
length(stopwords(source = "snowball"))
length(stopwords(source = "stopwords-iso"))
```

```
## [1] 571
## [1] 175
## [1] 1298
```

The length of these lists are quite varied, with the longest list being over seven times longer than the shortest! Let's examine the overlap of the words that appear in the three lists in Figure \@ref(fig:stopwordoverlap).

![(\#fig:stopwordoverlap)Set intersections for three common stop word lists](stopwords_files/figure-latex/stopwordoverlap-1.pdf) 

These three lists are almost true subsets of each other. The only excepetion is a set of ten words that appear in Snowball and ISO but not in the SMART list. What are those words?


```r
setdiff(
  stopwords(source = "snowball"),
  stopwords(source = "smart")
)
```

```
##  [1] "she's"   "he'd"    "she'd"   "he'll"   "she'll"  "shan't"  "mustn't"
##  [8] "when's"  "why's"   "how's"
```

All these words are contractions. This is *not* because the SMART lexicon doesn't include contractions, because if we look there are almost fifty of them.


```r
str_subset(stopwords(source = "smart"), "'")
```

```
##  [1] "a's"       "ain't"     "aren't"    "c'mon"     "c's"       "can't"    
##  [7] "couldn't"  "didn't"    "doesn't"   "don't"     "hadn't"    "hasn't"   
## [13] "haven't"   "he's"      "here's"    "i'd"       "i'll"      "i'm"      
## [19] "i've"      "isn't"     "it'd"      "it'll"     "it's"      "let's"    
## [25] "shouldn't" "t's"       "that's"    "there's"   "they'd"    "they'll"  
## [31] "they're"   "they've"   "wasn't"    "we'd"      "we'll"     "we're"    
## [37] "we've"     "weren't"   "what's"    "where's"   "who's"     "won't"    
## [43] "wouldn't"  "you'd"     "you'll"    "you're"    "you've"
```

We seem to have stumbled upon an inconsistency; why does SMART include `"he's"` but not `"she's"`? It is hard to say, but this would be worth rectifying before applying these stop word lists to an analysis or model preprocessing. It is likely that this stop word list was generated by selecting the most frequent words across a large corpus of text that had more representation for text about men than women. This is once again a reminder that we should always look carefully at any premade word list or other artifact we use to make sure it works well with our needs. 

\begin{rmdtip}
It is perfectly acceptable to start with a premade word list and remove
or append additional words according to your particular use case.
\end{rmdtip}


When you select a stop word list, it is important that you consider its size and breadth. Having a small and concise list of words can moderately reduce your token count while not having too great of an influence on your models, assuming that you picked appropriate words. As the size of your stop word list grows, each word added will have a diminishing positive effect with the increasing risk that a meaningful word has been placed on the list by mistake. In a later chapter on model building, we will show an example where we analyze the effects of different stop word lists.

### Stop word removal in R

Now that we have some stop word lists, we can move forward with removing these words. The particular way we remove stop words depends on the shape of our data. If you have your text in a tidy format with one word per row, you can use `filter()` from **dplyr** with a negated `%in%` if you have the stop words as a vector, or you can use `anti_join()` from **dplyr** if the stop words are in a `tibble()`. Like in our previous chapter, let's examine the text of "The Fir-Tree" by Hans Christian Andersen, and use **tidytext** to tokenize the text into words.


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
  unnest_tokens(word, text)
```

Let's use the Snowball stop word list as an example. Since the stop words return from this function as a vector, we will use `filter()`.


```r
tidy_fir_tree %>%
  filter(!(word %in% stopwords(source = "snowball")))
```

```
## # A tibble: 1,547 x 3
##    book         language word   
##    <chr>        <chr>    <chr>  
##  1 The fir tree English  far    
##  2 The fir tree English  forest 
##  3 The fir tree English  warm   
##  4 The fir tree English  sun    
##  5 The fir tree English  fresh  
##  6 The fir tree English  air    
##  7 The fir tree English  made   
##  8 The fir tree English  sweet  
##  9 The fir tree English  resting
## 10 The fir tree English  place  
## # ... with 1,537 more rows
```

If we use the `get_stopwords()` function from **tidytext** instead, then we can use the `anti_join()` function.


```r
tidy_fir_tree %>%
  anti_join(get_stopwords(source = "snowball"))
```

```
## # A tibble: 1,547 x 3
##    book         language word   
##    <chr>        <chr>    <chr>  
##  1 The fir tree English  far    
##  2 The fir tree English  forest 
##  3 The fir tree English  warm   
##  4 The fir tree English  sun    
##  5 The fir tree English  fresh  
##  6 The fir tree English  air    
##  7 The fir tree English  made   
##  8 The fir tree English  sweet  
##  9 The fir tree English  resting
## 10 The fir tree English  place  
## # ... with 1,537 more rows
```

The result of these two stop word removals is the same since we used the same stop word list in both cases.

## Creating your own stop words list

Another way to get a stop word list is to create one yourself. Let's explore a few different ways to find appropriate words to use. We will use the tokenized data from "The Fir-Tree" as a first example. Let's take the words and rank them by their count or frequency.

![(\#fig:unnamed-chunk-9)We counted words in "The Fir Tree" and ordered them by count or frequency.](stopwords_files/figure-latex/unnamed-chunk-9-1.pdf) 

We recognize many of what we would consider stop words in the first column here, with three big exceptions. We see `"tree"` at 3, `"fir"` at 12 and `"little"` at 22. These words appear high on our list but they do provide valuable information as they all reference the main character. What went wrong with this approach? Creating a stop word list using high-frequency words works best when it is created on a **corpus** of documents, not individual documents. This is because the words found in a single document will be document specific and the overall pattern of words will not generalize that well. 

\begin{rmdnote}
In NLP, a corpus is a set of texts or documents. The set of Hans
Christian Andersen's fairy tales can be considered a corpus, with each
fairy tale a document within that corpus. The set of United States
Supreme Court opinions can be considered a different corpus, with each
written opinion being a document within \emph{that} corpus.
\end{rmdnote}

The word `"tree"` does seem important as it is about the main character, but it could also be appearing so often that it stops providing any information. Let's try a different approach, extracting high-frequency words from the corpus of *all* English fairy tales by H.C. Andersen.

![(\#fig:unnamed-chunk-11)We counted words in all English fairy tales by Hans Christian Andersen and ordered them by count or frequency.](stopwords_files/figure-latex/unnamed-chunk-11-1.pdf) 

This list is more appropriate for our concept of stop words, and now it is time for us to make some choices. How many do we want to include in our stop word list? Which words should we add and/or remove based on prior information? Selecting the number of words to remove is best done by a case-by-case basis as it can be difficult to determine apriori how many different "meaningless" words appear in a corpus. Our suggestion is to start with a low number like twenty and increase by ten words until you get to words that are not appropriate as stop words for your analytical purpose. 

It is worth keeping in mind that this list is not perfect. It is based on the corpus of documents we had available, which is potentially biased since all the fairy tales were written by the same European white male from the early 1800s. 

\begin{rmdtip}
This bias can be minimized by removing words we would expect to be
over-represented or to add words we expect to be under-represented.
\end{rmdtip}

Easy examples are to include the compliments to the words in the lists if they are not present. Include `"big"` if `"small"` is present, `"old"` if `"young"` is present. This example list has words associated with women often listed lower in rank than words associated with men. With `"man"` being at rank 79 but `"woman"` at rank 179, choosing a threshold of 100 would lead to only one of these words being included. Depending on how important you think such nouns are going to be in your texts, either add `"woman"` or delete `"man"`.

Figure \@ref(fig:genderrank) shows how the words associated with men have higher rank than the words associated with women. By using a single threshold to create a stop word list, you would likely only include one form of such words.

![(\#fig:genderrank)We counted tokens and ranked according to total. Rank 1 has most occurrences.](stopwords_files/figure-latex/genderrank-1.pdf) 

Imagine now we would like to create a stop word list that spans multiple different genres, in such a way that the subject-specific stop words don't overlap. For this case, we would like words to be denoted as a stop word only if it is a stop word in all the genres. You could find the words individually in each genre and use the right intersections. However, that approach might take a substantial amount of time.

Below is a bad approach where we try to create a multi-language list of stop words. To accomplish this we calculate the [inverse document frequency](https://www.tidytextmining.com/tfidf.html) (IDF) of each word, and create the stop word list based on the words with the lowest IDF. The following function takes a tokenized dataframe and returns a dataframe with a column for each word and a column for the IDF.


```r
library(rlang)
calc_idf <- function(df, word, document) {
  words <- df %>%
    pull({{ word }}) %>%
    unique()

  n_docs <- length(unique(pull(df, {{ document }})))

  n_words <- df %>%
    nest(data = c({{ word }})) %>%
    pull(data) %>%
    map_dfc(~ words %in% unique(pull(.x, {{ word }}))) %>%
    rowSums()

  tibble(
    word = words,
    idf = log(n_docs / n_words)
  )
}
```

Here is the result where we try to create a cross-language list of stop words, by taking each fairy tale as a document. It is not very good! The overlap between what words appear in each language is very small, and that is what we mostly see in this list.

![(\#fig:unnamed-chunk-14)We counted words from all of H.C. Andersen's fairy tales in Danish, English, French, German, and Spanish and ordered by count or frequency.](stopwords_files/figure-latex/unnamed-chunk-14-1.pdf) 

TODO do same example with English only.

do MP, VP and  SAT 
https://pdfs.semanticscholar.org/c543/8e216071f6180c228cc557fb1d3c77edb3a3.pdf

## All stop word lists are context specific

Since context is so important in text modeling, it is important to make sure that the stop word list you use reflects the word space that you are planning on using it on. One common concern to consider is how pronouns bring information to your text. Pronouns are included in many different stop word lists (although inconsistently) and they will often *not* be noise in text data.

On the other hand, sometimes you will have to add in words yourself, depending on the domain. If you are working with texts for dessert recipes, certain ingredients (sugar, eggs, water) and actions (whisking, baking, stirring) may be frequent enough to pass your stop word threshold, but it's possible you will want to keep them as they may be informative. Throwing away "eggs" as a common word would make it harder or downright impossible to determine if certain recipes are vegan or not, while whisking and stirring may be fine to remove as distinguishing between recipes that do and don't require a whisk might not be that big of a deal.

## What happens when you remove stop words

We have discussed different ways of finding and removing stop words; now let's see what happens once you do remove them. First, let's explore the impact of the number of words that are included in the list. Figure \@ref(fig:stopwordresults) shows what percentage of words are removed as a function of the number of words in a text. The different colors represent the 3 different stop word lists we have considered in this chapter.

![(\#fig:stopwordresults)Proportion of words removed for different stop word lists and different document lengths](stopwords_files/figure-latex/stopwordresults-1.pdf) 

We notice, as we would predict, that larger stop word lists remove more words then shorter stop word lists. In this example with fairy tales, over half of the words have been removed, with the largest list removing over 80% of the words. We observe that shorter texts have a lower percentage of stop words. Since we are looking at fairy tales, this could be explained by the fact that a story has to be told regardless of the length of the fairy tale, so shorter texts are going to be more dense with more informative words.

Another problem you might have is dealing with misspellings. 

\begin{rmdwarning}
Most premade stop word lists assume that all the words are spelled
correctly.
\end{rmdwarning}

Handling misspellings when using premade lists can be done by manually adding common misspellings. You could imagine creating all words that are a certain string distance away from the stop words, but we do not recommend this as you would quickly include informative words this way.

One of the downsides of creating your own stop word lists using frequencies is that you are limited to using words that you have already observed. It could happen that `"she'd"` is included in your training corpus but the word `"he'd"` did not reach the threshold This is a case where you need to look at your words and adjust accordingly. Here the large premade stop word lists can serve as inspiration for missing words.

In a later chapter (TODO add link) we will investigate the influence of removing stop words in the context of modeling. Given the right list of words, you see no harm to the model performance, and may even see improvement in result due to noise reduction [@Feldman2007].

## Stop words in languages other than English

So far in this chapter, we have been spent the majority of the time on the English language, but English is not representative of every language. The stop word lists we examined in this chapter have been English and the notion of "short" and "long" lists we have used here are specific to English as a language. You should expect different languages to have a varying number of "uninformative" words, and for this number to depend on the morphological richness of a language; lists that contain all possible morphological variants of each stop word could become quite large.

Different languages have different numbers of words in each class of words. An example is how the grammatical case influences the articles used in German. Below are a couple of diagrams showing the use of definite and indefinite articles in German. Notice how German nouns have three genders (masculine, feminine, and neuter), which are not uncommon in languages around the world. Articles are almost always considered as stop words in English as they carry very little information. However, German articles give some indication of the case which can be used when selecting a list of stop words in German or any other language where the grammatical case is reflected in the text.

\captionsetup[table]{labelformat=empty,skip=1pt}
\begin{longtable}{lllll}
\caption*{
\large German Definite Articles (the)\\ 
\small \\ 
} \\ 
\toprule
 & Masculine & Feminine & Neuter & Plural \\ 
\midrule
Nominative & der & die & das & die \\ 
Accusative & den & die & das & die \\ 
Dative & dem & der & dem & den \\ 
Genitive & des & der & des & der \\ 
\bottomrule
\end{longtable}

\captionsetup[table]{labelformat=empty,skip=1pt}
\begin{longtable}{lllll}
\caption*{
\large German Indefinite Articles (a/an)\\ 
\small \\ 
} \\ 
\toprule
 & Masculine & Feminine & Neuter & Plural \\ 
\midrule
Nominative & ein & eine & ein & keine \\ 
Accusative & einen & eine & ein & keine \\ 
Dative & einem & einer & einem & keinen \\ 
Genitive & eines & einer & eines & keiner \\ 
\bottomrule
\end{longtable}


Building lists of stop words in Chinese has been done both manually and automatically [@Zou2006ACC] but so far none has been accepted as a standard [@Zou2006]. A full discussion of stop word identification in Chinese text would be out of scope for this book, so we will just highlight some of the challenges that differentiate it from English. 

\begin{rmdwarning}
Chinese text is much more complex than portrayed here. With different
systems and billions of users, there is much we won't be able to touch
on here.
\end{rmdwarning}

The main difference from English is the use of logograms instead of letters to convey information. However, Chinese characters should not be confused with Chinese words. The majority of words in modern Chinese are composed of multiple characters. This means that inferring the presence of words is more complicated and the notion of stop words will affect how this segmentation of characters is done.

## Summary

In many standard NLP work flows, the removal stop words is presented as a default or the correct choice without comment. Although removing stop words can improve the accuracy of your machine learning using text data, choices around such a step are complex. The content of existing stop word lists varies tremendously, and the available strategies for building your own can have subtle to not-so-subtle effects on your model results.

### In this chapter, you learned:

- what a stop word is and how to remove stop words from text data
- how different stop word lists can vary
- that the impact of stop word removal is different for different kinds of texts
- about the bias built in to stop word lists and strategies for building such lists
