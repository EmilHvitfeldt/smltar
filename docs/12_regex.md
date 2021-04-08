\cleardoublepage 

# (APPENDIX) Appendices {-}

# Regular expressions {#regexp}

> Some people, when confronted with a problem, think: "I know, I'll use regular expressions." Now they have two problems.  
> <footer>--- [Jamie Zawinski](https://en.wikiquote.org/wiki/Jamie_Zawinski)</footer>

This section will give a brief overview on how to write and use a [regular expression](https://en.wikipedia.org/wiki/Regular_expression), often abbreviated *regex*. Regular expressions are a way to specify or search for patterns of strings using a sequence of characters. By combining a selection of simple patterns, we can capture quite complicated strings.

Many functions in R take advantage of regular expressions. Some examples from base R include `grep`, `grepl`, `regexpr`, `gregexpr`, `sub`, `gsub`, and `strsplit`, as well as `ls` and `list.files`. The [**stringr**](https://github.com/tidyverse/stringr) package [@Wickham19] uses regular expressions extensively; the regular expressions are passed as the `pattern =` argument. Regular expressions can be used to detect, locate, or extract parts of a string.

## Literal characters

The most basic regular expression consists of only a single character. Here let's detect if each of the following strings in the character vector `animals` contains the letter "j".


```r
library(stringr)

animals <- c("jaguar", "jay", "bat")
str_detect(animals, "j")
```

```
#> [1]  TRUE  TRUE FALSE
```

We are also able to *extract* the match with `str_extract`. This may not seem too useful right now, but it becomes very helpful once we use more advanced regular expressions.


```r
str_extract(animals, "j")
```

```
#> [1] "j" "j" NA
```

Lastly we are able to *locate* the position of a match using `str_locate`.


```r
str_locate(animals, "j")
```

```
#>      start end
#> [1,]     1   1
#> [2,]     1   1
#> [3,]    NA  NA
```

<div class="rmdnote">
<p>The functions <code>str_detect</code>, <code>str_extract</code>, and <code>str_locate</code> are some of the most simple and powerful main functions in <strong>stringr</strong>, but the <strong>stringr</strong> package includes many more functions. To see the remaining functions, run <code>help(package = "stringr")</code> to open the documentation.</p>
</div>

We can also match multiple characters in a row.


```r
animals <- c("jaguar", "jay", "bat")
str_detect(animals, "jag")
```

```
#> [1]  TRUE FALSE FALSE
```

Notice how these characters are case sensitive.


```r
wows <- c("wow", "WoW", "WOW")
str_detect(wows, "wow")
```

```
#> [1]  TRUE FALSE FALSE
```

### Meta characters

There are 14 meta characters that carry special meaning inside regular expressions. We need to "escape" them with a backslash if we want to match the literal character (and backslashes need to be doubled in R). Think of "escaping" as stripping the character of its special meaning.

The plus symbol `+` is one of the special meta characters for regular expressions.


```r
math <- c("1 + 2", "14 + 5", "3 - 5")
str_detect(math, "\\+")
```

```
#> [1]  TRUE  TRUE FALSE
```

If we tried to use the plus sign without escaping it, like `"+"`, we would get an error and this line of code would not run.

The complete list of meta characters is displayed in Table \@ref(tab:metacharacters) [@theopengroup2018][@boost_c_libraries].


Table: (\#tab:metacharacters)All meta characters

|Description            |Character |
|:----------------------|:---------|
|opening square bracket |[         |
|closing square bracket |]         |
|backslash              |\\        |
|caret                  |^         |
|dollar sign            |$         |
|period/dot             |.         |
|vertical bar           |&#124;    |
|question mark          |?         |
|asterisk               |*         |
|plus sign              |+         |
|opening curly brackets |{         |
|closing curly brackets |}         |
|opening parentheses    |(         |
|closing parentheses    |)         |

## Full stop, the wildcard

Let's start with the full stop/period/dot, which acts as a "wildcard." This means that this character will match anything in place other then a newline character. 


```r
strings <- c("cat", "cut", "cue")
str_extract(strings, "c.")
```

```
#> [1] "ca" "cu" "cu"
```

```r
str_extract(strings, "c.t")
```

```
#> [1] "cat" "cut" NA
```

## Character classes

So far we have only been able to match either exact characters or wildcards. **Character classes** (also called character sets) let us do more than that. A character class allows us to match a character specified inside the class. A character class is constructed with square brackets. The character class `[ac]` will match *either* an "a" or a "c". 


```r
strings <- c("a", "b", "c")
str_detect(strings, "[ac]")
```

```
#> [1]  TRUE FALSE  TRUE
```

<div class="rmdnote">
<p>Spaces inside character classes are meaningful as they are interpreted as literal characters. Thus the character class <code>"[ac]"</code> will match the letter “a” and “c”, while the character class <code>"[a c]"</code> will match the letters “a” and “c” but also a space.</p>
</div>

We can use a hyphen character to define a range of characters. Thus `[1-5]` is the same as `[12345]`.


```r
numbers <- c("1", "2", "3", "4", "5", "6", "7", "8", "9")
str_detect(numbers, "[2-7]")
```

```
#> [1] FALSE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE FALSE FALSE
```

```r
sentence <- "This is a long sentence with 2 numbers with 1 digits."
str_locate_all(sentence, "[1-2a-b]")
```

```
#> [[1]]
#>      start end
#> [1,]     9   9
#> [2,]    30  30
#> [3,]    35  35
#> [4,]    45  45
```

We can also negate characters in a class with a caret `^`. Placing a caret immediately inside the opening square bracket will make the regular expression match anything *not* inside the class. Thus the regular expression `[^ac]` will match anything that isn't the letter "a" or "c".


```r
strings <- c("a", "b", "c")
str_detect(strings, "[^ac]")
```

```
#> [1] FALSE  TRUE FALSE
```

### Shorthand character classes

Certain character classes are so commonly used that they have been predefined with names. A couple of these character classes have even shorter shorthands. The class `[:digit:]` denotes all the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9 but it can also be described by `\\d`. Table \@ref(tab:characterclasses) presents these useful predefined character classes.


Table: (\#tab:characterclasses)All character classes

|Class              |Description                                                   |
|:------------------|:-------------------------------------------------------------|
|[:digit:] or \\\\d |Digits; [0-9]                                                 |
|[:alpha:]          |Alphabetic characters, uppercase and lowercase [A-z]          |
|[:alnum:]          |Alphanumeric characters, letters, and digits [A-z0-9]         |
|[:graph:]          |Graphical characters [[:alnum:][:punct:]]                     |
|[:print:]          |Printable characters [[:alnum:][:punct:][:space:]]            |
|[:lower:]          |Lowercase letters [a-z]                                       |
|[:upper:]          |Uppercase letters [A-Z]                                       |
|[:cntrl:]          |Control characters such as newline, carriage return, etc.     |
|[:punct:]          |Punctuation characters: !"#$%&’()*+,-./:;<=>?@[]^_`{&#124;}~  |
|[:blank:]          |Space and tab                                                 |
|[:space:] or \\\\s |Space, tab, vertical tab, newline, form feed, carriage return |
|[:xdigit:]         |Hexadecimal digits [0-9A-Fa-f]                                |
|\\\\S              |Not space [^[:space:]]                                        |
|\\\\w              |Word characters:  letters, digits, and underscores [A-z0-9_]  |
|\\\\W              |Non-word characters [^A-z0-9_]                                |
|\\\\D              |Non-digits [^0-9]                                             |

Notice that these short-hands are locale specific. This means that the danish character ø will be picked up in class `[:lower:]` but not in the class `[a-z]` as the character isn't located between a and z.

## Quantifiers

We can specify how many times we expect something to occur using quantifiers. If we want to find a digit with four numerals, we don't have to write `[:digit:][:digit:][:digit:][:digit:]`. Table \@ref(tab:greedyquantifiers) shows how to specify repetitions. Notice that `?` is shorthand for `{0,1}`, `*` is shorthand for `{0,}` and `+` is shorthand for `{1,}` [@javascriptinforegexpquantifiers].


Table: (\#tab:greedyquantifiers)Regular expression quantifiers

|Regex |Matches               |
|:-----|:---------------------|
|?     |zero or one times     |
|*     |zero or more times    |
|+     |one or more times     |
|{n}   |exactly n times       |
|{n,}  |at least n times      |
|{n,m} |between n and m times |

We can detect both color and colour by placing a quantifier after the "u" that detects 0 or 1 times used.


```r
col <- c("colour", "color", "farver")
str_detect(col, "colou?r")
```

```
#> [1]  TRUE  TRUE FALSE
```

And we can extract four-digit numbers using `{4}`.


```r
sentences <- c("The year was 1776.", "Alexander Hamilton died at 47.")
str_extract(sentences, "\\d{4}")
```

```
#> [1] "1776" NA
```

Sometimes we want the repetition to happen over multiple characters. This can be achieved by wrapping what we want repeated in parentheses. In the following example, we want to match all the instances of "NA" in the string. We put `"NA "` inside a set of parentheses and putting `+` after it to make sure we match at least once.


```r
batman <- "NA NA NA NA NA NA NA NA NA NA NA NA NA NA BATMAN!!!"
str_extract(batman, "(NA )+")
```

```
#> [1] "NA NA NA NA NA NA NA NA NA NA NA NA NA NA "
```

However, notice that this also matches the last space, which we don't want. We can fix this by matching zero or more "NA " followed by exactly 1 "NA".


```r
batman <- "NA NA NA NA NA NA NA NA NA NA NA NA NA NA BATMAN!!!"
str_extract(batman, "(NA )*(NA){1}")
```

```
#> [1] "NA NA NA NA NA NA NA NA NA NA NA NA NA NA"
```

By default these matches are "greedy", meaning that they will try to match the longest string possible. We can instead make them "lazy" by placing a `?` after, as shown in Table \@ref(tab:lazyquantifiers). This will make the regular expressions try to match the shortest string possible instead of the longest.


Table: (\#tab:lazyquantifiers)Lazy quantifiers

|regex  |matches                                                              |
|:------|:--------------------------------------------------------------------|
|??     |zero or one times, prefers 0                                         |
|*?     |zero or more times, match as few times as possible                   |
|+?     |one or more times, match as few times as possible                    |
|{n}?   |exactly n times, match as few times as possible                      |
|{n,}?  |at least n times, match as few times as possible                     |
|{n,m}? |between n and m times, match as few times as possible but at least n |

Comparing greedy and lazy matches gives us 3 and 7 "NA "'s respectively.


```r
batman <- "NA NA NA NA NA NA NA NA NA NA NA NA NA NA BATMAN!!!"
str_extract(batman, "(NA ){3,7}")
```

```
#> [1] "NA NA NA NA NA NA NA "
```

```r
str_extract(batman, "(NA ){3,7}?")
```

```
#> [1] "NA NA NA "
```

## Anchors

The meta characters `^` and `$` have special meaning in regular expressions. They force the engine to check the beginning and end of the string respectively, hence the name **anchor**. A mnemonic device to remember this is "First you get the power(`^`) and the you get the money(`\$`)".


```r
seasons <- c("The summer is hot this year",
             "The spring is a lovely time",
             "Winter is my favorite time of the year",
             "Fall is a time of peace")
str_detect(seasons, "^The")
```

```
#> [1]  TRUE  TRUE FALSE FALSE
```

```r
str_detect(seasons, "year$")
```

```
#> [1]  TRUE FALSE  TRUE FALSE
```

We can also combine the two to match a string completely.


```r
folder_names <- c("analysis", "data-raw", "data", "R")
str_detect(folder_names, "^data$")
```

```
#> [1] FALSE FALSE  TRUE FALSE
```

## Additional resources

This appendix covered some of the basics of getting started with (or refreshed about) regular expressions. If you want to learn more:

- RStudio maintains [an excellent collection of cheat sheets](https://www.rstudio.com/resources/cheatsheets/), some of which are related to regular expressions.
- www.rexegg.com has many pages of valuable information, including [this "quick start" page with helpful tables](https://www.rexegg.com/regex-quickstart.html).
- https://www.regular-expressions.info/ is another great general regular expression site.
- The [strings chapter](https://r4ds.had.co.nz/strings.html) in *R for Data Science* delves into examples written in R.
- Lastly if you want to go down to the metal, check out [*Mastering Regular Expressions*](http://shop.oreilly.com/product/9780596528126.do).

