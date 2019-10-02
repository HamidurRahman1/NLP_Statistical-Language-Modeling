
How to run:
    Just run the 'Main.py' file. It will do it's job and create answer for all questions in order.
    Answers are EXPLICITLY marked as there are functions defined for each of them.

Design and Explanation:
    Followed an OOP approach. PreProcess as an example does same pre-processing for TRAINING data and TEST data.
    At some point pre-processing TEST data depends on pre-processed TRAINING data that's why PreProcess takes optionally
    a pre-processed TRAINING object and do it's comparing.
    Same approach goes for Unigram, Bigram, BigramSmoothing (BigramAddOneSmoothing).

    'Util.py' defines some functions that are used by all classes.

    'QA.py' file consists of 7 functions from. Each function is named qa*(...) way meaning this function correspond to
     each question and generate an answer in a nicely format. Finally each function takes some necessary arguments to
     output an answer.

