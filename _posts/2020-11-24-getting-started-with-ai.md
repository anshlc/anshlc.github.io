---
layout: post
title: "Getting started with AI"
---

We'll be learning following fundamental concepts of AI with fastText:
1. what is fastText
2. Text classification
3. Training 
4. Validation
5. Tweaking the performance
6. Underfitting
7. Overfitting
8. Common performance metrics
9. How does it work?


## What is fastText
[fastText](https://fasttext.cc) is a free, open-source, lightweight library for efficient learning of word 
representations and sentence classification.

### Text Classification
Text classification is as fundamental as it is self-explanatory. It is essentially supervised learning. Applications 
of text classification range from spam filtering, sentiment analysis, content tagging/classification. It is also central
 to complex systems such as searching and ranking. Classification and categorization is the first step for humans and 
 machines to process texts in increasing volumes and complexity. In a vast universe of digital information, this is what
  allows you to locate your subject of interest, follow the echo chamber in which you’d like to occupy, etc.

### Training
**Sentiment analysis** is one popular text classification problem, which is basically the process of understanding if a 
given text is talking positively or negatively about a given subject. The dataset we are using is a set of millions of 
Amazon reviews posted by users.<br>
We need labeled data to train our supervised classifier. Hence some pre-processing is needed before training the model.
fasText expects each line of the text file contains a list of labels, followed by the corresponding document. All the 
labels should start by the `__label__` prefix, to distinguish between a label and a word.
For our dataset, all the reviews are divided into two classes, the classes are `__label__1` which means negative review
and `__label__2` meaning positive review. We have divided our dataset into two parts, for training and testing respectively.
Here is what the text file looks after pre-processing
{% highlight txt %}
$ head -n 3 train.txt
__label__2 Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well 
I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the 
games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate
guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^
__label__2 Whispers of the Wicked Saints: This was a easy to read book that made me want to keep reading on and on, not 
easy to put down.It left me wanting to read the follow on, which I hope is coming soon. I used to read a lot but have 
gotten away from it. This book made me want to read again. Very enjoyable.
__label__1 Buyer beware: This is a self-published book, and if you want to know why--read a few paragraphs! Those 5 star
reviews must have been written by Ms. Haddon's family and friends--or perhaps, by herself! I can't imagine anyone 
reading the whole thing--I spent an evening with the book and a friend and we were in hysterics reading bits and pieces
of it to one another. It is most definitely bad enough to be entered into some kind of a "worst book" contest. I can't
believe Amazon even sells this kind of thing. Maybe I can offer them my 8th grade term paper on "To Kill a 
Mockingbird"--a book I am quite sure Ms. Haddon never heard of. Anyway, unless you are in a mood to send a book to 
someone as a joke---stay far, far away from this one!
{% endhighlight %}
Now lets train our model on the dataset
{% highlight bash %}
$ ./fasttext supervised -input train.txt -output model_amzn
Read 289M words
Number of words:  5165173
Number of labels: 2
Progress: 100.0% words/sec/thread: 1707896 lr:  0.000000 avg.loss:  0.239325 ETA:   0h 0m 0s
{% endhighlight %}

At the end of training, a file model_cooking.bin, containing the trained classifier, is created in the current directory.
This model will be used for further validation and prediction.

### Validation
Now since the model is trained, lets test our classifier, fastText has a very simple command for this
{% highlight bash %}
$ ./fasttext predict model_amzn.bin -
> great product!
__label__2
> defective 
__label__1
{% endhighlight %}

To get a better sense of its quality, let's test it on the validation data by running:
{% highlight bash %}
$ ./fasttext test model_amzn.bin test.txt
N	400000
P@1	0.916
R@1	0.916
{% endhighlight %}

Here P@1 and R@1 are precision at one and recall at one
> *Precision* is the ratio of number of relevant results and total number of results retrieved by the program. 
>Assume a classifier, returned 3 labels out of which 2 are relevant to the word/sentence, then the precision is 
>2/3 (0.66). Since we have calculated the precision with 3 results, this is P@3.

> *Recall* is the ratio of relevant results retrieved by the algorithm and total number of the all relevant results. 
>With the same example above, if the total number of relevant labels of that word is 2, then the recall is 2 / 4.<br> 
>It is also termed as *Sensitivity*.<br>
>In mathematical terms:<br>
>Sensitivity = (True Positive)/(True Positive + False Negative)

> *Specificity* is defined as the proportion of actual negatives, which got predicted as the negative (or true negative)<br>
> Specificity = (True Negative)/(True Negative + False Positive)

### Tweaking the performance
The accuracy of our current model is ~90%, i.e 1 out of every 10 predictions can be wrong. A lot can be done to further 
improve the accuracy of our model. There are various parameters to tweak the performance like 
#### Learning rate
Learning rate of an algorithm indicates how much the model changes after each example sentence is processed. 
We can both increase and decrease the learning rate of an algorithm. A learning rate of 0 means that there is no 
change in learning, or the rate of change is just 0, so the model doesn’t change at all. 
A usual learning rate is 0.1 to 1.
<img src="/assets/lr.png" alt="Architecture">

#### epochs
Epoch is the number of times a model sees a phrase or an example input. By default, fasttext has epoch set to 5 i.e the 
model sees an example five times, increasing the epoch can help the model in learning better. But it can have reverse 
effect on the learning if this number is increased after a certain limit. This behaviour is called over-fitting which 
we will discuss later.<br>
Testing our newly trained model on validation dateset with varying epoch confirms this behaviour.
<img src="/assets/pre_vs_epoch.png" alt="Architecture">

As you can see, after epoch=10, the model performance starts deteriorating. Model starts over-fitting the training data,
 i.e instead of learning from data, it starts memorizing the data

#### Word n-grams
N-gram basically refer to the concatenation any n consecutive tokens. For eg. a 'unigram' refers to a single undividing 
unit, or token, usually used as an input to a model. For example a unigram can be a word or a letter depending on the 
model. In fastText, we work at the word level and thus unigrams are words. Similarly we denote by 'bigram' the concatenation 
of 2 consecutive tokens or words.
Training models with higher n-grams is usefull for classification problems where word order is important, such as 
sentiment analysis, which we are doing. Taking into account n words at a time allows model to store more context with a 
well-understood space–time tradeoff, which increases scalablity too.<br>
Lets see how precision changes when our model is tested with validation dateset with varying word n-grams value.
<img src="/assets/pr_vs_wordngrams.png" alt="Architecture">


### Underfitting
A statistical model or a machine learning algorithm is said to have underfitting when it cannot capture the underlying 
trend of the data. (It’s just like trying to fit undersized pants!) Underfitting destroys the accuracy of our machine 
learning model. Its occurrence simply means that our model or the algorithm does not fit the data well enough.

Techniques to reduce underfitting :
1. Increase model complexity
2. Increase number of features, performing feature engineering
3. Remove noise from the data.
4. Increase the number of epochs or increase the duration of training to get better results.

### Overfitting
Overfitting happens when a model starts to learn the details and noise in the training data to the extent that it 
negatively impacts the performance of the model on new data.
This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model.
The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize. 
Then the model does not categorize the data correctly, because of too many details and 
noise. The causes of overfitting are the non-parametric and non-linear methods because these types of machine learning 
algorithms have more freedom in building the model based on the dataset and therefore they can really build unrealistic 
models. A solution to avoid overfitting is using a linear algorithm if we have linear data or using the parameters like 
the maximal depth if we are using decision trees.

Techniques to reduce overfitting :
1. Increase training data.
2. Reduce model complexity.
3. Early stopping during the training phase (have an eye over the loss over the training period as soon as loss begins 
to increase stop training).
4. Use dropout for neural networks to tackle overfitting.

<img src="/assets/underfitting-overfitting.png" alt="Architecture">

## How does it work? Model training
Consider a dataset with following texts and labels
{% highlight js %}
__label__sports : India defeats Pakistan again!
__label__travel : Road trip
__label__sports : MSD retires
{% endhighlight %}

All the texts and labels are represented as vectors. We want to tweak the coordinates of these vectors so that the 
vectors for a given text and its associated label are very close to each other.
Consider a 2d example of the above dataset.
<img src="/assets/word_vectors.png" alt="Architecture">

Now how to tweak the vectors
1. Take the vector representing a text
2. Take the vector representing its label and put it in the function which returns a score
<img src="/assets/softmax.png" alt="Architecture">
3. Probabilities for all the labels need to be calculated
4. Calculate negative log likelihood and propogate the error.

Lets do the maths for our example dataset. Assume for the first text `India defeats Pakistan again!`, (text_1), the 
scoring functions returns following values <br>
{% highlight js %}
__label__sports : -2
__label__travel : -5
{% endhighlight %}
The scoring function can be as simple as `s = - distance(label, text)`

Now lets put these scores in the softmax function
{% highlight js %}
__label__sports : 0.95
__label__travel : 0.04
{% endhighlight %}

Propogating the loss :  given the probabilities of both the labels, we can calculate negative log likelihood scores 
{% highlight js %}
loss =  -log(p(correct label)) 

loss_sports = 0.02227639471
loss_travel = 1.39794000867
{% endhighlight %}
Now if the model predicts __`label__sports` for text_1, the loss is low (0.02227639471). This is expected as the model 
has predicted the right label. On the other side, if the model would have predicted `__loss_travel`, there is something 
wrong with our model, and high loss value confirms the same. 


Now this method is very computationaly expensive, we need to calculate probabiltities for each text with each label.
fastText uses a different method called *Hierarchical softmax*



## References
1. [fastText](https://fasttext.cc/docs/en/supervised-tutorial.html)
2. [Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)
3. [Underfitting and Overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/)
4. [arXiv:1607.04606v2 [cs.CL]](https://arxiv.org/abs/1607.04606)



