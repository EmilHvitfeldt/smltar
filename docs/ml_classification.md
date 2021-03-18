# Classification



What is classification?

## First attempt

The first attempt and full game will use the same data. 
The first attempt might use a subset of the data to make the example easier to understand.
and properly give a balanced dataset, which then later can be explored.

### Look at the data {#classfirstattemptlookatdata}

We are going to be working with US consumer complaints on financial products and company responses.
It contains a text field containing the complaint along with information regarding what it was for,
how it was filed and the response. 
In this chapter, we will try to predict what type of product the complaints are referring to. 
This first attempt will be limited to predicting if the product is a mortgage or not.

We can read in the complaint data \@ref(us-consumer-finance-complaints) with `read_csv()`.


```r
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stringr)
library(discrim)
library(readr)

complaints <- read_csv("data/complaints.csv.gz")
```

then we will start by taking a quick look at the data to see what we have to work with


```r
glimpse(complaints)
```

```
## Rows: 117,214
## Columns: 18
## $ date_received                <date> 2019-09-24, 2019-10-25, 2019-11-08, 2...
## $ product                      <chr> "Debt collection", "Credit reporting, ...
## $ sub_product                  <chr> "I do not know", "Credit reporting", "...
## $ issue                        <chr> "Attempts to collect debt not owed", "...
## $ sub_issue                    <chr> "Debt is not yours", "Information belo...
## $ consumer_complaint_narrative <chr> "transworld systems inc. \nis trying t...
## $ company_public_response      <chr> NA, "Company has responded to the cons...
## $ company                      <chr> "TRANSWORLD SYSTEMS INC", "TRANSUNION ...
## $ state                        <chr> "FL", "CA", "NC", "RI", "FL", "TX", "S...
## $ zip_code                     <chr> "335XX", "937XX", "275XX", "029XX", "3...
## $ tags                         <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA...
## $ consumer_consent_provided    <chr> "Consent provided", "Consent provided"...
## $ submitted_via                <chr> "Web", "Web", "Web", "Web", "Web", "We...
## $ date_sent_to_company         <date> 2019-09-24, 2019-10-25, 2019-11-08, 2...
## $ company_response_to_consumer <chr> "Closed with explanation", "Closed wit...
## $ timely_response              <chr> "Yes", "Yes", "Yes", "Yes", "Yes", "Ye...
## $ consumer_disputed            <chr> "N/A", "N/A", "N/A", "N/A", "N/A", "N/...
## $ complaint_id                 <dbl> 3384392, 3417821, 3433198, 3366475, 33...
```

The first thing to note is our target variable `product` which we need to trim only display "Mortgage" and "Other",
and the `consumer_complaint_narrative` variable which contains the complaints.
Here is the first 6 complaints:


```r
head(complaints$consumer_complaint_narrative)
```

```
## [1] "transworld systems inc. \nis trying to collect a debt that is not mine, not owed and is inaccurate."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
## [2] "I would like to request the suppression of the following items from my credit report, which are the result of my falling victim to identity theft. This information does not relate to [ transactions that I have made/accounts that I have opened ], as the attached supporting documentation can attest. As such, it should be blocked from appearing on my credit report pursuant to section 605B of the Fair Credit Reporting Act."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
## [3] "Over the past 2 weeks, I have been receiving excessive amounts of telephone calls from the company listed in this complaint. The calls occur between XXXX XXXX and XXXX XXXX to my cell and at my job. The company does not have the right to harass me at work and I want this to stop. It is extremely distracting to be told 5 times a day that I have a call from this collection agency while at work."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
## [4] "I was sold access to an event digitally, of which I have all the screenshots to detail the transactions, transferred the money and was provided with only a fake of a ticket. I have reported this to paypal and it was for the amount of {$21.00} including a {$1.00} fee from paypal. \n\nThis occured on XX/XX/2019, by paypal user who gave two accounts : 1 ) XXXX 2 ) XXXX XXXX"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
## [5] "While checking my credit report I noticed three collections by a company called ARS that i was unfamiliar with. I disputed these collections with XXXX, and XXXX and they both replied that they contacted the creditor and the creditor verified the debt so I asked for proof which both bureaus replied that they are not required to prove anything. I then mailed a certified letter to ARS requesting proof of the debts n the form of an original aggrement, or a proof of a right to the debt, or even so much as the process as to how the bill was calculated, to which I was simply replied a letter for each collection claim that listed my name an account number and an amount with no other information to verify the debts after I sent a clear notice to provide me evidence. Afterwards I recontacted both XXXX, and XXXX, to redispute on the premise that it is not my debt if evidence can not be drawn up, I feel as if I am being personally victimized by ARS on my credit report for debts that are not owed to them or any party for that matter, and I feel discouraged that the credit bureaus who control many aspects of my personal finances are so negligent about my information."
## [6] "I would like the credit bureau to correct my XXXX XXXX XXXX XXXX balance. My correct balance is XXXX"
```

Throughout the narratives is a series of capital x's. This has been done to hide Personally identifiable information (PII). This is not a universal censoring mechanism and can vary from source to source, hopefully you will be able to get this information in the data dictionary but you should always look at the data yourself to verify. We also see that all monetary amounts are surrounded by curly brackets, this is another step of preprocessing that has been done for us.

We can craft a regular expression to extract all the dollar amounts.


```r
complaints$consumer_complaint_narrative %>%
  str_extract_all("\\{\\$[0-9\\.]*\\}") %>%
  compact() %>%
  head()
```

```
## [[1]]
## [1] "{$21.00}" "{$1.00}" 
## 
## [[2]]
## [1] "{$2300.00}"
## 
## [[3]]
## [1] "{$200.00}"  "{$5000.00}" "{$5000.00}" "{$770.00}"  "{$800.00}" 
## [6] "{$5000.00}"
## 
## [[4]]
## [1] "{$15000.00}" "{$11000.00}" "{$420.00}"   "{$15000.00}"
## 
## [[5]]
## [1] "{$0.00}" "{$0.00}" "{$0.00}" "{$0.00}"
## 
## [[6]]
## [1] "{$650.00}"
```

## Building our first classification model

Since this data is given to us after the fact, 
we need to make sure that only the information that would be available at the time of prediction is included in the model,
otherwise we are going to be very disappointed once the model is pushed to production.
The variables we can use as predictors are

- `date_received`
- `issue`
- `sub_issue`
- `consumer_complaint_narrative`
- `company`
- `state`
- `zip_code`
- `tags`
- `submitted_via`

Many of these have quite a lot of levels.
First we will include `date_received` for further consideration, along with `consumer_complaint_narrative` and `tags`.
`submitted_via` would have been a viable candidate too, but all the entries are "web".
The other variables could be of use too, but they are categorical variables with many values so we will exclude them for now.

We start by splitting the data into a training and testing dataset.
But before we do that we will create a factor variable of `product` with the levels "Mortgage" and "Other".
Then we will use the `initial_split()` from **rsample** to create a binary split of the data. 
The `strata` argument is used to make sure that the split is created to make sure the distribution of `product` is similar in the training set and testing set. 
Since the split is done using random sampling we set a seed so we can reproduce the results.


```r
set.seed(1234)
complaints2class <- complaints %>%
  mutate(product = factor(if_else(product == "Mortgage", "Mortgage", "Other")))

complaints_split <- initial_split(complaints2class, strata = product)

complaints_train <- training(complaints_split)
complaints_test <- testing(complaints_split)
```

Looking at the dimensions of the two split shows that it worked successfully.


```r
dim(complaints_train)
```

```
## [1] 87911    18
```

```r
dim(complaints_test)
```

```
## [1] 29303    18
```

Next we need to do some preprocessing. We need to do this since the models we are trying to use only support all numeric data. 

\begin{rmdnote}
Some models are able to handle factor variables and missing data. But it
is in our best interest to manually deal with these problems so we know
how they are handled.
\end{rmdnote}

The **recipes** package allows us to create a specification of the preprocessing steps we want to perform. Furthermore it contains the transformations we have trained on the training set and apply them in the same way for the testing set.
First off we use the `recipe()` function to initialize a recipe, we use a formula expression to specify the variables we are using along with the dataset.


```r
complaints_rec <-
  recipe(product ~ date_received + tags + consumer_complaint_narrative,
    data = complaints_train
  )
```

First will we take a look at the `date_received` variable. We use the `step_date()` to extract the month and day of the week (dow). Then we remove the original variable and dummify the variables created with `step_dummy()`.


```r
complaints_rec <- complaints_rec %>%
  step_date(date_received, features = c("month", "dow"), role = "dates") %>%
  step_rm(date_received) %>%
  step_dummy(has_role("dates"))
```

the `tags` variable includes some missing data. We deal with this by using `step_unknown()` to that adds a new level to the factor variable for cases of missing data. Then we dummify the variable with `step_dummy()`


```r
complaints_rec <- complaints_rec %>%
  step_unknown(tags) %>%
  step_dummy(tags)
```

Lastly we use **textrecipes** to handle the `consumer_complaint_narrative` variable. First we perform tokenization to words with `step_tokenize()`, by default this is done using `tokenizers::tokenize_words()`.
Next we remove stopwords with `step_stopwords()`, the default choice is the snowball stopword list, but custom lists can be provided too. Before we calculate the tf-idf we use `step_tokenfilter()` to only keep the 50 most frequent tokens, this is to avoid creating too many variables. To end off we use `step_tfidf()` to perform tf-idf calculations.


```r
complaints_rec <- complaints_rec %>%
  step_tokenize(consumer_complaint_narrative) %>%
  step_stopwords(consumer_complaint_narrative) %>%
  step_tokenfilter(consumer_complaint_narrative, max_tokens = 50) %>%
  step_tfidf(consumer_complaint_narrative)
```

Now that we have a full specification of the recipe we run `prep()` on it to train each of the steps on the training data.


```r
complaint_prep <- prep(complaints_rec)
```

We can now extract the transformed training data with `juice()`. To apply the prepped recipe to the testing set we use the `bake()` function.


```r
train_data <- juice(complaint_prep)
test_data <- bake(complaint_prep, complaints_test)
```

For the modeling we will use a simple Naive Bayes model (TODO add citation to both Naive Bayes and its use in text classification).
One of the main advantages of Naive Bayes is its ability to handle a large number of features that we tend to get when using word count methods.
Here we have only kept the 50 most frequent tokens, we could have kept more tokens and a Naive Bayes model would be able to handle it okay, but we will limit it for this first time.


```r
nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("klaR")
nb_spec
```

```
## Naive Bayes Model Specification (classification)
## 
## Computational engine: klaR
```

Now we have everything we need to fit our first classification model, we just have to run `fit()` on our model specification and our training data.


```r
nb_fit <- nb_spec %>%
  fit(product ~ ., data = train_data)
```

We have more successfully fitted out first classification model.

### Evaluation

One option for our evaluating our model is to predict one time on the test set to measure performance. The test set is extremely valuable data, however, and in real world situations, you can only use this precious resource one time (or at most, twice). The purpose of the test data is to estimate how your final model will perform on new data. Often during the process of modeling, we want to compare models or different model parameters. We can't use the test set for this; instead we use **resampling**.

For example, let's estimate the performance of the Naive Bayes classification model we just fit. We can do this using resampled datasets built from the training set. Let's create cross 10-fold cross-validation sets, and use these resampled sets for performance estimates.


```r
complaints_folds <- vfold_cv(complaints_train)

complaints_folds
```

```
## #  10-fold cross-validation 
## # A tibble: 10 x 2
##    splits               id    
##    <named list>         <chr> 
##  1 <split [79.1K/8.8K]> Fold01
##  2 <split [79.1K/8.8K]> Fold02
##  3 <split [79.1K/8.8K]> Fold03
##  4 <split [79.1K/8.8K]> Fold04
##  5 <split [79.1K/8.8K]> Fold05
##  6 <split [79.1K/8.8K]> Fold06
##  7 <split [79.1K/8.8K]> Fold07
##  8 <split [79.1K/8.8K]> Fold08
##  9 <split [79.1K/8.8K]> Fold09
## 10 <split [79.1K/8.8K]> Fold10
```

Each of these "splits" contains information about how to create cross-validation folds from the original training data. In this example, 90% of the training data is included in each fold and the other 10% is held out for evaluation.

For convenience, let's use `workflows()` for our resampling estimates of performance. These are convenience functions that fit different modeling functions like recipes, model specifications, etc. together so they are easier to pass around in a modeling project.


```r
nb_wf <- workflow() %>%
  add_recipe(complaints_rec) %>%
  add_model(nb_spec)

nb_wf
```

```
## == Workflow ==================================================================================================================
## Preprocessor: Recipe
## Model: naive_Bayes()
## 
## -- Preprocessor --------------------------------------------------------------------------------------------------------------
## 9 Recipe Steps
## 
## * step_date()
## * step_rm()
## * step_dummy()
## * step_unknown()
## * step_dummy()
## * step_tokenize()
## * step_stopwords()
## * step_tokenfilter()
## * step_tfidf()
## 
## -- Model ---------------------------------------------------------------------------------------------------------------------
## Naive Bayes Model Specification (classification)
## 
## Computational engine: klaR
```

In the last section, we fit one time to the training data as a whole. Now, to estimate how well that model performs, let's fit the model many times, once to each of these resampled folds, and then evaluate on the heldout part of each resampled fold.


```r
nb_rs <- fit_resamples(
  nb_wf,
  complaints_folds
)

nb_rs
```

```
## #  10-fold cross-validation 
## # A tibble: 10 x 4
##    splits               id     .metrics         .notes          
##    <list>               <chr>  <list>           <list>          
##  1 <split [79.1K/8.8K]> Fold01 <tibble [2 x 3]> <tibble [0 x 1]>
##  2 <split [79.1K/8.8K]> Fold02 <tibble [2 x 3]> <tibble [0 x 1]>
##  3 <split [79.1K/8.8K]> Fold03 <tibble [2 x 3]> <tibble [0 x 1]>
##  4 <split [79.1K/8.8K]> Fold04 <tibble [2 x 3]> <tibble [0 x 1]>
##  5 <split [79.1K/8.8K]> Fold05 <tibble [2 x 3]> <tibble [0 x 1]>
##  6 <split [79.1K/8.8K]> Fold06 <tibble [2 x 3]> <tibble [0 x 1]>
##  7 <split [79.1K/8.8K]> Fold07 <tibble [2 x 3]> <tibble [0 x 1]>
##  8 <split [79.1K/8.8K]> Fold08 <tibble [2 x 3]> <tibble [0 x 1]>
##  9 <split [79.1K/8.8K]> Fold09 <tibble [2 x 3]> <tibble [0 x 1]>
## 10 <split [79.1K/8.8K]> Fold10 <tibble [2 x 3]> <tibble [0 x 1]>
```

What results do we see, in terms of performance metrics?


```r
nb_rs %>%
  collect_metrics()
```

```
## # A tibble: 2 x 5
##   .metric  .estimator  mean     n std_err
##   <chr>    <chr>      <dbl> <int>   <dbl>
## 1 accuracy binary     0.882    10 0.00593
## 2 roc_auc  binary     0.927    10 0.00220
```

## Different types of models

(Not all of these models are good, but are used to show strengths and weaknesses)

- SVM
- Naive Bayes
- glmnet
- Random forrest
- knn
- NULL model

## Two class or multiclass

## Case study: What happens if you don't censor your data

The complaints data already have sensitive information censored out with XXXX and XX.
This can be seen as a kind of annotation, we don't get to know the specific account numbers and birthday which would be mostly unique anyways and filtered out.

Below we have is the most frequent trigrams [#tokenizing-by-n-grams] from our training dataset.


```r
complaints_train %>%
  slice(1:1000) %>%
  unnest_tokens(trigrams, consumer_complaint_narrative,
    token = "ngrams",
    collapse = FALSE
  ) %>%
  count(trigrams, sort = TRUE) %>%
  mutate(censored = str_detect(trigrams, "xx")) %>%
  slice(1:20) %>%
  ggplot(aes(n, reorder(trigrams, n), fill = censored)) +
  geom_col() +
  scale_fill_manual(values = c("grey40", "firebrick")) +
  labs(y = "Trigrams", x = "Count")
```

![(\#fig:censoredtrigram)Many of the most frequent trigrams feature censored words.](ml_classification_files/figure-latex/censoredtrigram-1.pdf) 

As you see the vast majority includes one or more censored words.
Not only does the most used trigrams include some kind of censoring, 
but the censored words include some signal as they are not used uniformly between the products.
In the following chart, we take the top 25 most frequent trigrams that includes one of more censoring,
and plot the proportions of the usage in "Mortgage" and "Other".


```r
top_censored_trigrams <- complaints_train %>%
  slice(1:1000) %>%
  unnest_tokens(trigrams, consumer_complaint_narrative,
    token = "ngrams",
    collapse = FALSE
  ) %>%
  count(trigrams, sort = TRUE) %>%
  filter(str_detect(trigrams, "xx")) %>%
  slice(1:25)

plot_data <- complaints_train %>%
  unnest_tokens(trigrams, consumer_complaint_narrative,
    token = "ngrams",
    collapse = FALSE
  ) %>%
  right_join(top_censored_trigrams, by = "trigrams") %>%
  count(trigrams, product, .drop = FALSE)

plot_data %>%
  ggplot(aes(n, trigrams, fill = product)) +
  geom_col(position = "fill")
```

![](ml_classification_files/figure-latex/trigram25-1.pdf)<!-- --> 

There is a good spread in the proportions, tokens like "on xx xx" and "of xx xx" are used when referencing to a date, eg "we had a problem on 06/25 2012".
Remember that the current tokenization engine strips the punctuation before tokenizing. 
This means that the above examples are being turned into "we had a problem on 06 25 2012" before creating n-grams.

We can as a practical example replace all cases of XX and XXXX with random integers to crudely simulate what the data might look like before it was censored. 
This is going a bit overboard since dates will be given values between 00 and 99 which would not be right, 
and that we don't know if only numerics have been censored.
Below is a simple function `uncesor_vec()` that locates all instances of `XX` and replaces them with a number between 11 and 99.
We don't need to handle the special case of `XXXX` as it automatically being handled.


```r
uncensor <- function(n) {
  as.character(sample(seq(10^(n - 1), 10^n - 1), 1))
}

uncensor_vec <- function(x) {
  locs <- str_locate_all(x, "XX")

  map2_chr(x, locs, ~ {
    for (i in seq_len(nrow(.y))) {
      str_sub(.x, .y[i, 1], .y[i, 2]) <- uncensor(2)
    }
    .x
  })
}
```

And we can run a quick test to see if it works.


```r
uncensor_vec("In XX/XX/XXXX I leased a XXXX vehicle")
```

```
## [1] "In 78/49/3119 I leased a 2759 vehicle"
```

Now we try to produce the same chart as \@ref(fig:censoredtrigram) but with the only difference being that we apply our uncensoring function to the text before tokenizing.


```r
complaints_train %>%
  slice(1:1000) %>%
  mutate(text = uncensor_vec(consumer_complaint_narrative)) %>%
  unnest_tokens(trigrams, text,
    token = "ngrams",
    collapse = FALSE
  ) %>%
  count(trigrams, sort = TRUE) %>%
  mutate(censored = str_detect(trigrams, "xx")) %>%
  slice(1:20) %>%
  ggplot(aes(n, reorder(trigrams, n), fill = censored)) +
  geom_col() +
  scale_fill_manual(values = c("grey40", "firebrick")) +
  labs(y = "Trigrams", x = "Count")
```

![(\#fig:uncensoredtrigram)Trigrams without numbers flout to the top as the uncensored tokens are too spread out.](ml_classification_files/figure-latex/uncensoredtrigram-1.pdf) 

The same trigrams that appear in the last chart appeared in this one as well, 
but none of the uncensored words appear in the top which is what is to be expected.
This is expected because while `xx xx 2019` appears towards the top in the first as it indicates a date in the year 2019, having that uncensored would split it into 365 buckets.
Having dates being censored gives more power to pick up the signal of a date as a general construct giving it a higher chance of being important.
But it also blinds us to the possibility that certain dates and months are more prevalent.

We have talked a lot about censoring data in this section.
Another way to look at this is a form of preprocessing in your data pipeline.
It is very unlikely that you want any specific person's social security number, credit card number or any other kind of personally identifiable information ([PII](https://en.wikipedia.org/wiki/Personal_data)) imbedded into your model.
Not only is it likely to provide a useful signal as they appear so rarely and most likely highly correlated with other known variables in your database.
More importantly, that information can become embedded in your model and begin to leak if you are not careful as showcased by @carlini2018secret, @Fredrikson2014 and @Fredrikson2015.
Both of these issues are important, and one of them could land you in a lot of legal trouble if you are not careful. 

If for example, you have a lot of social security numbers you should definitely not pass them on to your model, but there is no hard in annotation the presence of a social security number. 
Since a social security number has a very specific form we can easily construct a regular expression \@ref(regexp) to locate them.

\begin{rmdnote}
A social security number comes in the form AAA-BB-CCCC where AAA is a
number between 001 and 899 excluding 666, BB is a number between 01 and
99 and CCCC is a number between 0001 and 9999. This gives us the
following regex

(?!000\textbar666){[}0-8{]}{[}0-9{]}\{2\}-(?!00){[}0-9{]}\{2\}-(?!0000){[}0-9{]}\{4\}
\end{rmdnote}

We can use a replace function to replace it with something that can be picked up by later preprocessing steps. 
A good idea is to replace it with a "word" that won't be accidentally broken up by a tokenizer.


```r
ssn_text <- c(
  "My social security number is 498-08-6333",
  "No way, mine is 362-60-9159",
  "My parents numbers are 575-32-6985 and 576-36-5202"
)

ssn_pattern <- "(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}"

str_replace_all(
  string = ssn_text,
  pattern = ssn_pattern,
  replacement = "ssnindicator"
)
```

```
## [1] "My social security number is ssnindicator"           
## [2] "No way, mine is ssnindicator"                        
## [3] "My parents numbers are ssnindicator and ssnindicator"
```

This technique isn't just useful for personally identifiable information but can be used anytime you want to intentionally but similar words in the same bucket, hashtags, emails, and usernames can sometimes also benefit from being annotated.

## Case study: Adding custom features

Most of what we have looked at so far have boiled down to counting occurrences of tokens and weighting them in one way or another.
This approach is quite broad and domain agnostic so it might miss some important parts.
Having domain knowledge over your data allows you to extract, hopefully, more powerful, features from the data that wouldn't come up in the naive search from simple tokens.
As long as you can reasonably formulate what you are trying to count, chances are you can write a function that can detect it.
This is where having a little bit of @regexp pays off.

\begin{rmdnote}
A noteable package is
\href{https://github.com/mkearney/textfeatures}{textfeatures} which
includes many functions to extract all different kinds of metrics.
textfeatures can be used in textrecipes with the
\texttt{step\_textfeature()} function.
\end{rmdnote}

If you have some domain knowledge you might know something that can provide a meaningful signal.
It can be simple things like; the number of URLs and the number of punctuation marks.
But it can also be more tailored such as; the percentage of capitalization, does the text end with a hashtag, or are two people's names both mentioned in this text.

It is clear by looking at the data, that certain patterns repeat that have not adequately been picked up by our model so far.
These are related to the censoring and the annotation regarding monetary amounts that we saw in [#classfirstattemptlookatdata].
In this section, we will walk through how to create functions to extract the following features

- Detect credit cards
- Calculate percentage censoring
- Detect monetary amounts

### Detecting credit cards

We know that the credit card is represented as 4 groups of 4 capital Xs.
Since the data is fairly well processed we are fairly sure that spacing will not be an issue and all credit cards will be represented as "XXXX XXXX XXXX XXXX". 
The first naive attempt is to use str_detect with "XXXX XXXX XXXX XXXX" to find all the credit cards.
It is a good idea to create a small example where you know the answer then prototyping your functions before moving them to the main data.
We start by creating a vector with 2 positives, 1 negative and 1 potential false positive.
The last string is more tricky since it has the same shape as a credit card but has one too many groups.


```r
credit_cards <- c(
  "my XXXX XXXX XXXX XXXX balance, and XXXX XXXX XXXX XXXX.",
  "card with number XXXX XXXX XXXX XXXX.",
  "at XX/XX 2019 my first",
  "live at XXXX XXXX XXXX XXXX XXXX SC"
)


str_detect(credit_cards, "XXXX XXXX XXXX XXXX")
```

```
## [1]  TRUE  TRUE FALSE  TRUE
```

And we see what we feared, the last vector got falsely detected to be a credit card.
Sometimes you will have to accept a certain number of false positives and false negatives depending on the data and what you are trying to detect. 
In this case, we can make the regex a little more complicated to avoid that specific false positive.
We need to make sure that the word coming before the X's doesn't end in a capital X and the word following the last X doesn't start with a capital X.
We place spaces around the credit card and use some negated character classes[#character-classes] to detect anything BUT a capital X.


```r
str_detect(credit_cards, "[^X] XXXX XXXX XXXX XXXX [^X]")
```

```
## [1]  TRUE FALSE FALSE FALSE
```

Hurray! This fixed the false positive. 
But it gave us a false negative in return.
Turns out that this regex doesn't allow the credit card to be followed by a period since it requires a space.
We can fix this with an alternation to match for a period or a space and a non X.


```r
str_detect(credit_cards, "[^X] +XXXX XXXX XXXX XXXX(\\.| [^X])")
```

```
## [1]  TRUE  TRUE FALSE FALSE
```

Know that we have a regular expression we are happy with we can turn it into a function we can use.
We can extract the presence of a credit card with `str_detect()` and the number of credit cards with `str_count()`.


```r
creditcard_indicator <- function(x) {
  str_detect(x, "[^X] +XXXX XXXX XXXX XXXX(\\.| [^X])")
}

creditcard_count <- function(x) {
  str_count(x, "[^X] +XXXX XXXX XXXX XXXX(\\.| [^X])")
}

creditcard_indicator(credit_cards)
```

```
## [1]  TRUE  TRUE FALSE FALSE
```

```r
creditcard_count(credit_cards)
```

```
## [1] 2 1 0 0
```

### Calculate percentage censoring

Some of the complaints contain quite a lot of censoring, and we will try to extract the percentage of the text that is censored.
There are often many ways to get to the same solution when working with regular expressions.
I will attack this problem by counting the number of X's in each string, then count the number of alphanumeric characters and divide the two to get a percentage.


```r
str_count(credit_cards, "X")
```

```
## [1] 32 16  4 20
```

```r
str_count(credit_cards, "[:alnum:]")
```

```
## [1] 44 30 17 28
```

```r
str_count(credit_cards, "X") / str_count(credit_cards, "[:alnum:]")
```

```
## [1] 0.7272727 0.5333333 0.2352941 0.7142857
```

And we finish up by creating a function.


```r
procent_censoring <- function(x) {
  str_count(x, "X") / str_count(x, "[:alnum:]")
}

procent_censoring(credit_cards)
```

```
## [1] 0.7272727 0.5333333 0.2352941 0.7142857
```

### Detecting monetary amounts

We have already constructed a regular expression that detects the monetary amount from the text.
So we can look at how we can use this information.
Let us start by creating a little example and see what we can extract.


```r
dollar_texts <- c(
  "That will be {$20.00}",
  "{$3.00}, {$2.00} and {$7.00}",
  "I have no money"
)

str_extract_all(dollar_texts, "\\{\\$[0-9\\.]*\\}")
```

```
## [[1]]
## [1] "{$20.00}"
## 
## [[2]]
## [1] "{$3.00}" "{$2.00}" "{$7.00}"
## 
## [[3]]
## character(0)
```

We can create a function that simply detects the dollar amount, and we can count the number of times each amount appears.
But since each occurrence also has a value, would it be nice to include that information as well, such as the mean, minimum or maximum.

First, let's extract the number from the strings, we could write a regular expression for this, but the `parse_number()` function from the readr package does a really good job of pulling out numbers.


```r
str_extract_all(dollar_texts, "\\{\\$[0-9\\.]*\\}") %>%
  map(readr::parse_number)
```

```
## [[1]]
## [1] 20
## 
## [[2]]
## [1] 3 2 7
## 
## [[3]]
## numeric(0)
```

Now that we have the number we can iterate over them with the function of our choice.
Since we are going to have texts with no amounts we need to make sure that we have to handle the case with zero numbers. Defaults for some functions with length 0 vectors can before undesirable as we don't want `-Inf` to be a value. I'm going to extract the maximum value and will denote cases with no values to have a maximum of 0.


```r
max_money <- function(x) {
  str_extract_all(x, "\\{\\$[0-9\\.]*\\}") %>%
    map(readr::parse_number) %>%
    map_dbl(~ ifelse(length(.x) == 0, 0, max(.x)))
}

max_money(dollar_texts)
```

```
## [1] 20  7  0
```

know that we have created some feature extraction functions we can use them to hopefully make our classification model better.

## Case Study: feature hashing

## What evaluation metrics are appropriate

Data will most likely be sparse when using BoW

## Full game

### Feature selection

### Splitting the data

### Specifying models

### Cross-validation

### Evaluation

Interpretability.

"Can we get comparable performance with a simpler model?"
Compare with simple rule-based model
