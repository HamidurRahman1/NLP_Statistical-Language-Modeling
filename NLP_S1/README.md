
<strong>What it is: </strong><br>
<pre>A language model based on statistics learns the likelihood of word event dependent on instances of content. More
straightforward models may take a gander at a setting of a short grouping of words, while bigger models may work at the
degree of sentences or sections. Most ordinarily, language models work at the degree of words.</pre>

<strong>Models: </strong><br>
<li>Unigram</li>
<li>Bigram</li>
<li>Bigram with Add-One smoothing</li>
<br>

<strong>How to run: </strong><br>
<pre>Just run the 'Main.py' file. It will do it's job and create answer for all questions in order.
Answers are EXPLICITLY marked as there are functions defined for each of them.</pre>

<strong>Design and Explanation:</strong>
<pre>Followed an OOP approach. PreProcess as an example does same pre-processing for TRAINING data and
TEST data. At some point pre-processing TEST data depends on pre-processed TRAINING data that's why
PreProcess takes optionally a pre-processed TRAINING object and do it's comparing.
Same approach goes for Unigram, Bigram, BigramSmoothing (BigramAddOneSmoothing).

'Util.py' defines some functions that are used by all classes.

'QA.py' file consists of 7 functions. Each function is named qa*(...) meaning this function
correspond to each question and generate an answer in a nicely format and before all these each
function of course takes some necessary arguments to output an answer.</pre>
