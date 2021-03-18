# (PART) Natural Language Features {-}

# Language and modeling {#language}



Machine learning and deep learning models for text are put into action by computers, but they are designed and trained by human beings. As natural language processing (NLP) practitioners, we bring our assumptions about what language is and how language works into the task of building language models. This is true *even when* we don't think about how language works very deeply or when our understanding is unsophisticated or inaccurate; speaking a language is not the same as understanding how it works. We can improve our machine learning models for text by heightening that understanding.

Throughout the course of this book, we will discuss these kinds of improvements and how they are related to language. Data scientists involved in the everyday work of text analysis and text modeling typically don't have formal training in how language works, but there is an entire field focused on exactly that, **linguistics**.

## Linguistics for text analysis

@Briscoe13 provides helpful introductions to what linguistics is and how it intersects with the practical computational field of natural language processing. The broad field of linguistics includes subfields focusing on different aspects of language, which are somewhat hierarchical, as shown in Table \@ref(tab:lingsubfields).


\begin{table}

\caption{(\#tab:lingsubfields)Some subfields of linguistics, moving from smaller structures to broader structures}
\centering
\begin{tabular}[t]{l|l}
\hline
Linguistics subfield & What does it focus on?\\
\hline
Phonetics & Sounds that people use in language\\
\hline
Phonology & Systems of sounds in particular languages\\
\hline
Morphology & How words are formed\\
\hline
Syntax & How sentences are formed from words\\
\hline
Semantics & What sentences mean\\
\hline
Pragmatics & How language is used in context\\
\hline
\end{tabular}
\end{table}

These fields each study a different level at which language exhibits organization. At the same time, this organization and the rules of language can be ambiguous. Beatrice Santorini, a linguist at the University of Pennsylvania, compiles examples of just such ambiguity from [news headlines](https://www.ling.upenn.edu/~beatrice/humor/headlines.html):

> Include Your Children When Baking Cookies

> March Planned For Next August

> Enraged Cow Injures Farmer with Ax

> Wives Kill Most Spouses In Chicago

If you don't have knowledge about what linguists study and what they know about language, these news headlines are just hilarious. To linguists, these are hilarious *because they exhibit certain kinds of semantic ambiguity*.

Notice also that the first two subfields on this list are about sounds, i.e., speech. Most linguists view speech as primary, and writing down language as text as a technological step.

\begin{rmdnote}
Remember that some language is signed, not spoken, so the description
laid out here is limited.
\end{rmdnote}

Written text is typically less creative and further from the primary language than we would wish. This points out how fundamentally limited modeling from written text is. Imagine the abstract language data we want exists in some high-dimensional latent space; we would like to extract that information using the text somehow, but it just isn't completely possible. Any model we build is inherently limited.


## A glimpse into one area: morphology

How can a deeper knowledge of how language works inform text modeling? Let's focus on **morphology**, the study of words' internal structures and how they are formed, to illustrate this. Words are medium to small in length in English; English has a moderately low ratio of morphemes (the smallest unit of language with meaning) to words while other languages like Turkish and Russian have a higher ratio of morphemes to words [@Bender13]. A related idea is the categorization of languages as either more analytic (like Mandarin or modern English, breaking up concepts into separate words) or synthetic (like Hungarian or Swahili, combining concepts into one word). 

Morphology focuses on how morphemes such as prefixes, suffixes, and root words come together to form words. However, even the very question of what a word is turns out to be difficult, and not only for languages other than English. Compound words in English like "real estate" and "dining room" represent one concept but contain whitespace. The morphological characteristics of a text dataset are deeply connected to preprocessing steps like tokenization (Chapter \@ref(tokenization)), removing stop words (Chapter \@ref(stopwords)), and even stemming (Chapter \@ref(stemming)). These preprocessing steps, in turn, have dramatic effects on model results.

the Danish alphabet is on the surface similar to the one used in English with the addition of 3 more letters æ, ø, and å which comes at the end of the alphabet. However, Danish doesn't use all the remaining letters natively. C, q, w, x, and z are only used in words adopted from other languages giving Danish a higher ratio of vowels to consonants since æ, ø, and å are all vowels. Despite this does Danish have over 25 phonetically distinct vowel sounds [@konig2002germanic] which along with other language-specific features makes it hard to infer spelling from speech alone. One noticeable difference coming from English to Danish is the use of compound nouns. Words such as "brandbil" (fire truck), "politibil" (police car) and "lastbil" (truck) all start with the morpheme "bil" (car) and prefixes with denoting the type of car. Additionally, some nouns will because of this seem more descriptive then their English counterpart, "vaskebjørn" (raccoon) splits into the morphemes "vaske" "bjørn" literally meaning "washing bear", when working with Danish and other languages with compound words, such as German, it might provide benefits to split to apply compound splitting to extract more information [@Sugisaki2018].

## Different languages

We believe that most of the readers of this book are probably native English speakers, and most of the text used in training machine learning models is also English. However, English is by no means a dominant language globally, especially as a native or first language. As an example close to home for us, of the two authors of this book, one is a native English speaker and one is not. According to the [comprehensive and detailed Ethnologue project](https://www.ethnologue.com/language/eng), less than 20% of the world's population speaks English at all.

@Bender11 provides guidance to computational linguists building models for text, for any language. One specific point she makes is to name the language being studied.

> **Do** state the name of the language that is being studied, even if it's English. Acknowledging that we are working on a particular language foregrounds the possibility that the techniques may in fact be language-specific. Conversely, neglecting to state that the particular data used were in, say, English, gives [a] false veneer of language-independence to the work. 

This idea is simple (acknowledge that the models we build are typically language-specific) but the [#BenderRule](https://twitter.com/search?q=%23BenderRule) has led to increased awareness of the limitations of the current state of this field. Our book is not geared toward academic NLP researchers developing new methods, but toward data scientists and analysts working with everyday datasets; this issue is relevant even for us. [Name the languages used in training models](https://thegradient.pub/the-benderrule-on-naming-the-languages-we-study-and-why-it-matters/), and think through what that means for their generalizability. We will practice what we preach and tell you that most of the text used for modeling in this book is English, with some text in Danish. 

## Other ways text can vary

The concept of differences in language is relevant for modeling beyond only the broadest language level (for example, English vs. Danish vs. German vs. Farsi). Language from a specific dialect often cannot be handled well with a model trained on data from the same language but not inclusive of that dialect. One dialect used in the United States is African American Vernacular English (AAVE). Models trained to detect toxic or hate speech are more likely to falsely identify AAVE as hate speech [@Sap19]; this is deeply troubling not only because the model is less accurate than it should be, but because it amplifies harm against an already marginalized group.

Language is also changing over time. This is a known characteristic of language; if you notice the evolution of your own language, don't be depressed or angry, because it means that people are using it! Teenage girls are especially effective at language innovation, and have been for centuries [@McCulloch15]; innovations spread from groups such as young women to other parts of society. This is another difference that impacts modeling.

\begin{rmdtip}
Differences in language relevant for models also include the use of
slang, and even the context or medium of that text.
\end{rmdtip}

Consider two bodies of text, both mostly standard written English, but one made up of tweets and one made up of legal documents. If an NLP practitioner trains a model on the dataset of tweets to predict some characteristics of the text, it is very possible (in fact, likely, in our experience) that the model will perform poorly if applied to the dataset of legal documents. Like machine learning in general, text modeling is exquisitely sensitive to the data used for training. This is why we are somewhat skeptical of AI products such as sentiment analysis APIs, not because they *never* work well, but because they work well only when the text you need to predict from is a good match to the text such a product was trained on.

## Summary 

Linguistics is the study of how language works, and while we don't believe real-world NLP practitioners must be experts in linguistics, learning from such domain experts can improve both the accuracy of our models and our understanding of why they do (or don't!) perform well. Predictive models for text reflect the characteristics of their training data, so differences in language over time, between dialects, and in various cultural contexts can prevent a model trained on one data set from being appropriate for application in another. A large amount of the text modeling literature focuses on English, but English is not a dominant language around the world.


### In this chapter, you learned:

- that areas of linguistics focus on topics from sounds to how language is used
- how a topic like morphology is connected to text modeling steps
- to identify the language you are modeling, even if it is English
- about many ways language can vary and how this can impact model results
