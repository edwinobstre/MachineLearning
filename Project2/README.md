# MachineLearning Project2

### Abstract 
This is a  project to build a review prediction. Use the bag-of-words and word-embedding methods to deal with the raw reviews from yelp.com, imdb.com and amazon.com. Select SVM, adaboost, logistic regression and Navie Bayes algorithm to train the model.

### How to use
python(>=3.3)
pandas package 
matplotlib package
numpy package
sklearn package
keras package

cd ./code.py
python3 code.py

note: dataset should be in the same directory .

### Dataset
This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

-Format:
review \t score \n
Score is either 1 (for positive) or 0 (for negative)    

It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

The reviews come from three different websites/fields:

imdb.com
amazon.com
yelp.com



