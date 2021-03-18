# Data {#appendixdata}

There are four main text data sets we use throughout this book to demonstrate building features for machine learning and training models. These data sets include texts of different languages, different lengths (short to long), and from very recent time periods to a few hundred years ago. 

These text data sets are not overly difficult to read into memory and prepare for analysis, but in many text modeling projects, the data itself may be in any of a number of formats from an API to literal paper. Practitioners may need to use skills such as web scraping or connecting to databases to even begin their work.

## H.C. Andersen fairy tales

The **hcandersenr** [@R-hcandersenr] package includes the text of the 157 known fairy tales by the Danish author H.C. Andersen. 
There are five different languages available, with:

- 156 fairy tales in English,
- 154 in Spanish,
- 150 in German,
- 138 in Danish, and
- 58 in French.

The package contains a data set for each language with the naming convention `hcandersen_**`,
where `**` is a country code.
Each data set comes as a dataframe with two columns, `text` and `book` where the `book` variable has the text divided into strings of up to 80 characters.

The package also makes available a data set called `EK` which includes information about the publication date, language of origin, and names in the different languages.

## Consumer Financial Protection Bureau (CFPB) complaints 

Consumers can submit complaints to the [United States Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/) about financial products and services; the CFPB sends the complaints to companies for response. The data set of consumer complaints used in this book has been filtered to those submitted to the CFPB since 1 January 2019 that include a consumer complaint narrative (i.e., some submitted text). Each observation has a `complaint_id`, various categorical variables, and a text column `consumer_complaint_narrative` containing the written complaints.


## Opinions of the Supreme Court of the United States 

The **scotus** [@R-scotus] package contains a sample of the Supreme Court of the United States' opinions.
The `scotus_sample` dataframe includes one opinion per row along with the year, case name, docket number, and a unique ID number.

The text has had minimal preprocessing and includes header information in the text field, such as shown here:
 

```
#> No. 97-1992
#> VAUGHN L. MURPHY, Petitioner v. UNITED PARCEL SERVICE, INC.
#> ON WRIT OF CERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR THE TENTH
#> CIRCUIT
#> [June 22, 1999]
#> Justice O'Connor delivered the opinion of the Court.
#> Respondent United Parcel Service, Inc. (UPS), dismissed petitioner Vaughn
#> L. Murphy from his job as a UPS mechanic because of his high blood pressure.
#> Petitioner filed suit under Title I of the Americans with Disabilities Act of
#> 1990 (ADA or Act), 104 Stat. 328, 42 U.S.C. § 12101 et seq., in Federal District
#> Court. The District Court granted summary judgment to respondent, and the Court
#> of Appeals for the Tenth Circuit affirmed. We must decide whether the Court
#> of Appeals correctly considered petitioner in his medicated state when it held
#> that petitioner's impairment does not "substantially limi[t]" one or more of
#> his major life activities and whether it correctly determined that petitioner
#> is not "regarded as disabled." See §12102(2). In light of our decision in Sutton
#> v. United Air Lines, Inc., ante, p. ____, we conclude that the Court of Appeals'
#> resolution of both issues was correct.
```

## Kickstarter campaign blurbs

This data set includes tktk...

