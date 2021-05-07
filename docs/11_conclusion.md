\cleardoublepage 

# (PART) Conclusion {-}

# Text models in the real world {-}

Models affect real people in real ways. As the school year of 2020 began with many schools in the United States operating online only because of the novel coronavirus pandemic, a parent of a junior high student [reported](https://twitter.com/DanaJSimmons/status/1300639757165191170) that her son was deeply upset and filled with doubt because of the way the algorithm of an ed tech company automatically scored\index{automated grading} his text answers. The parent and child discovered [@Verge2020] how to "game" the ed tech system's scoring.

> Algorithm update. He cracked it: Two full sentences, followed by a word salad of all possibly applicable keywords. 100% on every assignment. Students on \@EdgenuityInc, there's your ticket. He went from an F to an A+ without learning a thing.

We can't know the details of the proprietary modeling and/or heuristics that make up the ed tech system's scoring algorithm, but there is enough detail in this student's experience to draw some conclusions. We surmise that this is a count-based method or model, perhaps a linear one but not necessarily so. The success of "word salad" submissions indicates that the model or heuristic being applied has not learned that complex, or even correct, language is important for the score.

What could a team building this kind of score do to avoid these problems? It seems like "word salad" type submissions were not included in the training data as negative examples (i.e., with low scores), indicating that the training data was _biased_; it did not reflect\index{bias} the full spectrum of submissions that the system sees in real life. The system (code and data) is not auditable for teachers or students, and the ed tech company does not directly have a process in place to handle appeals or mistakes in the score itself. 

The particular ed tech company in this example does claim that these scores are used only to provide scoring guidance to teachers and that teachers can either accept or overrule such scores, but it is not clear how often teachers overrule scores. This highlights the foundational question about whether such a model or system should even be built to start with; with its current performance, this system is failing at what educators and students understand its goals to be, and is doing harm to its users. 

This situation is more urgent and important than only a single example from the pandemic-stressed United States educational system, because:

- these types of harms exacerbate existing inequalities, and

- these systems are becoming more and more widely used.

@Ramineni2018 report how GRE essays by African-American students receive lower scores from automatic grading algorithms\index{automated grading} than from expert human graders, and explore statistical differences in the two grading approaches. This is a stark reminder that machine learning systems learn patterns from training data and amplify those patterns. @Feathers2019 reports that the kind of automatic essay grading described here is used in at least 21 states, and essay grading is not the only kind of predictive text model that has real impact on real individuals' lives^[For more, see this discussion from Rachel Thomas: https://youtu.be/bqCEUQq0z4o]. 

As you finish this book and take away ideas on how to transform language to features for modeling and how to build reliable text models, we want to end by reflecting on how our work as data practitioners plays out when applied. Language data is richly human, and what you and we do with it matters.

