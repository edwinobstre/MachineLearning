# MachineLearning Project1

### Abstract 
This is a  project to build a logistic regression model without sklearn package, the code fulfills the SGD optimization, cross-validation hyperparameter selection and model prediction analysis. The project also plots the relationship between hyperparameters and accuracy based on titanic and mnist dataset.

### How to use
python(>=3.3)
pandas package 
matplotlib package
autograd package
numpy package

cd ./titanic_classifier.py
python3 titanic_classifier.py
cd ./mnist_classifier.py
python3 mnist_classifier.py

note: please put the titanic_train and mnist_train under the same directory.

### Dataset
The titanic.csv file contains data for 887 of the real Titanic passengers. Each row represents one person. The columns describe different attributes about the person including whether they survived (S), their age (A), their passenger-class (C), their sex (G) and the fare they paid (X).

[more information about titanic](http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

[more information about mnist](http://yann.lecun.com/exdb/mnist/)

[get the csv dataset](https://pjreddie.com/projects/mnist-in-csv/)

### Report
loss with iterations(mnist dataset)

<img src="tree/master/Project1/project_picture/mnist_SGD_iteration.jpg" alt="alt text" width=500 height=350>

loss with time(mnist dataset)

<img src="tree/master/Project1/project_picture/mnist_SGD_loss_time.jpg" alt="alt text" width=500 height=350>

hyperparameter with accuracy(titanic)
lambda chosen = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]

<img src="tree/master/Project1/project_picture/accuracy-titanic.jpg" alt="alt text" width=500 height=350>

hyperparameter with accuracy(mnist)

<img src="tree/master/Project1/project_picture/mnist_accuracy.jpg" alt="alt text" width=500 height=350>


