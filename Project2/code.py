# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


#Module
class MyClassifier:
    def __init__(self, options):
        """
        constructor of the MyClassifier
        options[0]: the train data_file name
        options[1]: the test data_file_name
        options[2]: the pre-train data processing method choices, ['bag-of-words','word-embedding']
        options[3]: the model selection,['SVM', 'adaboost', 'Logistic Regression','Naive Bayes']
        options[4]: the sentences(document) length for pad the encode
        options[5]: the train_test_ratio for the train data
        hyper_para is the best hyper-parameter for the model, the default value is 0
        X is the features of train data after pre-train
        y is the labels of train data after pre-train
        :param options
        """
        self.datafile_name_train = options[0]
        self.datafile_name_test = options[1]
        self.pre_train_chosen = options[2]
        self.model_chosen = options[3]
        self.document_len = options[4]
        self.train_test_ratio = options[5]
        self.hyper_para = 0
        self.X = 0
        self.y = 0

    def read_data_bag_of_words_function(self, datafile_name):
        """
        This function is for reading txt files in
        It will strip all the useless  infomation in the txt file such as punctuation, numeric values.
        It will also convert upper case to lower case and use bag-of-words to transfer document to vector.
        """
        X = []
        Y = []
        # open the train file and strip all punctuation and numeric values, change uppercase to lowercase
        with open(self.datafile_name_train, "r") as f:
            train_list = []
            for line in f:
                digit = [s for s in line if s.isdigit()]
                Y += [int(digit[-1])]
                line = re.sub('[' + string.punctuation + ']', '', line)
                line = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", line)
                lines = line.lower()
                train_list.append(lines)
        # open the test file and strip all punctuation and numeric values, change uppercase to lowercase
        with open(self.datafile_name_test, "r") as f1:
            test_list = []
            for line1 in f1:
                line1 = re.sub('[' + string.punctuation + ']', '', line1)
                line1 = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", line1)
                line1 = line1.lower()
                test_list.append(line1)
        # get the labels from the train data.
        Y = numpy.asarray(Y)
        # use CountVectorizer and TfidfTransformer
        vectorizer = CountVectorizer(lowercase=True, stop_words='english')
        tf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        X_train_counts = vectorizer.fit_transform(train_list)
        X_test_counts = vectorizer.transform(test_list)
        X_train_tf = tf_transformer.fit_transform(X_train_counts).toarray()
        X_test_tf = tf_transformer.transform(X_test_counts).toarray()
        # if the model is 'Naive Bayes', binary the features
        if self.model_chosen == 'Naive Bayes' or self.model_chosen == 'adaboost':
            X_binary = binarize(X_train_tf)
            for i in X_binary:
                temp = [int(j) for j in i]
                X += [temp]
            X_train_tf = numpy.asarray(X)
        # if the datafile_name is train data, bag-of-words function return [X, y]
        if datafile_name == self.datafile_name_train:
            return [X_train_tf, Y]
        # if the datafile_name is test data, bag-of-words function return X
        elif datafile_name == self.datafile_name_test:
            return X_test_tf

    def read_data_embedding_function(self, datafile_name):
        """
        This function is for reading data.txt files in
        It will exclude all the useless  information in the data.txt file such as punctuation, numeric values.
        It will also convert upper case to lower case and use word-embedding to transfer document to vector.
        """
        Y = []
        # open the train file and change uppercase to lowercase, exclude the numeric and punctuation
        with open(self.datafile_name_train) as f1:
            train_data = []
            for line1 in f1:
                digit = [s for s in line1 if s.isdigit()]
                Y += [int(digit[-1])]
                line1 = re.sub('[' + string.punctuation + ']', '', line1)
                line1 = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", line1)
                lines1 = line1.lower()
                train_data.append(lines1)

        # open the test file and change uppercase to lowercase, exclude the numeric and punctuation
        with open(self.datafile_name_test) as f2:
            test_data = []
            for line2 in f2:
                line2 = re.sub('[' + string.punctuation + ']', '', line2)
                line2 = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", line2)
                lines2 = line2.lower()
                test_data.append(lines2)
        # get the labels from train data
        Y = numpy.asarray(Y)
        # if datafile name is train data, return [X, Y]
        if datafile_name == self.datafile_name_train:
            # prepare tokenizer
            t = Tokenizer()
            t.fit_on_texts(train_data)
            vocab_size = len(t.word_index) + 1
            # integer encode the documents
            encoded_X = t.texts_to_sequences(train_data)
            # pad documents to a max length of documents words
            padded_docs = pad_sequences(encoded_X, maxlen=self.document_len, padding='post')
            # load the whole embedding into memory
            embeddings_index = dict()
            f = open('glove.6B.50d.txt')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = numpy.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
            f.close()
            # create a weight matrix for words in train data
            embedding_matrix = numpy.zeros((vocab_size, 50))
            for word, i in t.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            # sum up row vectors
            X = numpy.zeros((len(Y),50))
            dict1 = numpy.array(padded_docs)
            for i in range(len(dict1)):
                for j in range(len(dict1[0])):
                    X[i, :] = numpy.add(X[i, :], embedding_matrix[dict1[i, j], :])
            return [X, Y]
        # if the datafile name is test data
        elif  datafile_name == self.datafile_name_test:
            # prepare tokenizer
            t = Tokenizer()
            t.fit_on_texts(test_data)
            vocab_size = len(t.word_index) + 1
            # integer encode the documents
            encoded_X = t.texts_to_sequences(test_data)
            # pad documents to a max length of documents words
            padded_docs = pad_sequences(encoded_X, maxlen=self.document_len, padding='post')
            # load the whole embedding into memory
            embeddings_index = dict()
            f = open('glove.6B.50d.txt')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = numpy.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
            f.close()
            # create a weight matrix for words in test data
            embedding_matrix = numpy.zeros((vocab_size, 50))
            for word, i in t.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            # sum up each row vectors
            X = numpy.zeros((len(test_data), 50))
            dict1 = numpy.array(padded_docs)
            for i in range(len(dict1)):
                for j in range(len(dict1[0])):
                    X[i, :] = numpy.add(X[i, :], embedding_matrix[dict1[i, j], :])
            return X

    def split_data(self, X, Y):
        """
        This function used for spliting data into training and testing data.
        :param X: features of train data
        :param Y: label of train data
        :return: X_train, y_train, X_test, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = self.train_test_ratio)
        return [X_train, X_test, y_train, y_test]

    def fit_choose_hyperparamter(self):
        """
        this function used for choosing the pre-train processing method and choosing the train model
        then using cross-validation, according accuracy with uncertainty,select the best hyper-parameter
        for the model report the precision, recall, f1-score, support and each score's min, max and average value
        :return:
        """
        train_error = []
        # pre-train method choosing
        if self.pre_train_chosen == 'bag-of-words':
            self.X, self.y = self.read_data_bag_of_words_function(self.datafile_name_train)
        elif self.pre_train_chosen == 'word-embedding':
            self.X, self.y = self.read_data_embedding_function(self.datafile_name_train)
        # call split data function
        X_train, X_test, y_train, y_test = self.split_data(self.X, self.y)
        # model is SVM, hyper-parameter chosen [1, 10, 100, 1000]
        if self.model_chosen == 'SVM' and self.pre_train_chosen == 'bag-of-words':
            # Set the parameters by cross-validation
            tuned_parameters = [{'kernel': ['linear'], 'gamma': [0.7], 'C': [1, 10, 100, 1000]}]
            scores = ['precision', 'recall']
            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                clf = GridSearchCV(SVC(), tuned_parameters, cv=5, refit=True,
                                   scoring='%s_macro' % score)
                clf.fit(X_train, y_train)
                print("Best parameters set found on development set:")
                print()
                self.hyper_para = clf.best_params_
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))
                    train_error.append(1.0-mean)
                print()
                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, clf.predict(X_test)
                target_names = ['label0', 'label1']
                print(classification_report(y_true, y_pred, target_names=target_names))

                print()
            # plot train error and validation error
            validation_error = train_error[4:]
            train_error = train_error[:4]
            labels = ["Train_error", "validation_error"]
            hyper_para = [1, 10, 100, 1000]
            fig, ax = plt.subplots()
            train_plot = ax.plot(hyper_para, train_error, 'o-', label='Train_error')
            valid_plot = ax.plot(hyper_para, validation_error, 'o-', label='validation_error')
            plt.xlabel('hyper-parameters:C')
            plt.ylabel('error')
            plt.title('SVM train_validation_error  with different C')
            plt.legend()
            plt.show()

        # model is adaboost, hyper-parameters chosen is [20, 30, 50 ,100]
        elif self.model_chosen == 'adaboost':
            param_grid = {"base_estimator__criterion": ["entropy"],
                          "base_estimator__splitter": ["best"],
                          "n_estimators": [20, 30, 50, 100]}
            scores = ['precision', 'recall']
            dtc = tree.DecisionTreeClassifier(random_state=11, max_features="auto", max_depth=None)
            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()
                # run grid search
                abc = GridSearchCV(AdaBoostClassifier(base_estimator=dtc), param_grid=param_grid, scoring='roc_auc')
                abc.fit(X_train, y_train)
                print("Best parameters set found on development set:")
                print()
                self.hyper_para = abc.best_params_
                print(abc.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = abc.cv_results_['mean_test_score']
                stds = abc.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, abc.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))
                    train_error.append(1.0 - mean)
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, abc.predict(X_test)
                target_names = ['label0', 'label1']
                print(classification_report(y_true, y_pred, target_names=target_names))
                print()
            validation_error = train_error[4:]
            train_error = train_error[:4]
            labels = ["Train_error", "validation_error"]
            hyper_para = [20, 30, 50, 100]
            fig, ax = plt.subplots()
            ax.plot(hyper_para, train_error, 'o-', label='Train_error')
            ax.plot(hyper_para, validation_error, 'o-', label='validation_error')
            plt.xlabel('hyper-parameters:n_estimators')
            plt.ylabel('error')
            plt.title('Adaboost train_validation_error  with different number of estimators')
            plt.legend()
            plt.show()

        # model is Logistic Regression, hyper-parameters chosen [1, 10, 100, 1000]
        elif self.model_chosen == 'Logistic Regression':
            # Set the parameters by cross-validation
            tuned_parameters = [{'C': [1, 10, 100, 1000]}]
            scores = ['precision', 'recall']
            for score in scores:
                logreg = GridSearchCV(LogisticRegression(penalty='l2'), tuned_parameters, refit=True,
                                      scoring='%s_macro' % score)
                logreg.fit(X_train, y_train)
                print("Best parameters set found on development set:")
                print()
                self.hyper_para = logreg.best_params_
                print(logreg.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = logreg.cv_results_['mean_test_score']
                stds = logreg.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, logreg.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                    train_error.append(1.0 - mean)
                print()
                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, logreg.predict(X_test)
                target_names = ['label0', 'label1']
                print(classification_report(y_true, y_pred, target_names=target_names))
                print()
            validation_error = train_error[4:]
            train_error = train_error[:4]
            labels = ["Train_error", "validation_error"]
            hyper_para = [1, 10, 100, 1000]
            fig, ax = plt.subplots()
            ax.plot(hyper_para, train_error, 'o-', label='Train_error')
            ax.plot(hyper_para, validation_error, 'o-', label='validation_error')
            plt.xlabel('hyper-parameters:lambda')
            plt.ylabel('error')
            plt.title('Logistic Regression train_validation_error  with different lambda')
            plt.legend()
            #plt.show()

        # model is Naive Bayes, hyper-parameters is chosen [1 , 5 , 10, 20]
        elif self.model_chosen == 'Naive Bayes' and self.pre_train_chosen == 'bag-of-words':
            # Set the parameters by cross-validation
            tuned_parameters = [{'alpha': [1, 5, 10, 20]}]
            scores = ['precision', 'recall']
            for score in scores:
                nb = GridSearchCV(MultinomialNB(), tuned_parameters, refit=True,
                                      scoring='%s_macro' % score)
                nb.fit(X_train, y_train)
                print("Best parameters set found on development set:")
                print()
                self.hyper_para = nb.best_params_
                print(nb.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = nb.cv_results_['mean_test_score']
                stds = nb.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, nb.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                    train_error.append(1.0 -mean)
                print()
                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, nb.predict(X_test)
                target_names = ['label0', 'label1']
                print(classification_report(y_true, y_pred, target_names=target_names))
                print()
            validation_error = train_error[4:]
            train_error = train_error[:4]
            labels = ["Train_error", "validation_error"]
            hyper_para = [1, 5, 10, 20]
            fig, ax = plt.subplots()
            ax.plot(hyper_para, train_error, 'o-', label='Train_error')
            ax.plot(hyper_para, validation_error, 'o-', label='validation_error')
            plt.xlabel('hyper-parameters:smoothing')
            plt.ylabel('error')
            plt.title('Naive Bayes train_validation_error  with different smooth')
            plt.legend()
            plt.show()


    def test_evaluation_function(self):
        X = []
        # pre-train chosen for test data
        if self.pre_train_chosen == 'bag-of-words':
            X = self.read_data_bag_of_words_function(self.datafile_name_test)
        elif self.pre_train_chosen == 'word-embedding':
            X = self.read_data_embedding_function(self.datafile_name_test)
        # test model is SVM, if pre-train is word-embedding, it is time-consuming method
        # user should change another model or pre-train method
        if self.model_chosen == 'SVM':
            if self.pre_train_chosen == 'word-embedding':
                print("It's time consuming for svm model using word-embedding method, change another model")
            else:
                optimal_svm = SVC()
                optimal_svm.set_params(**self.hyper_para)
                optimal_svm.fit(self.X, self.y)
                y_pred = optimal_svm.predict(X)
                print(y_pred)
        # test model is adaboost
        elif self.model_chosen == 'adaboost':
            DTC = tree.DecisionTreeClassifier(random_state=11, max_features="auto", max_depth=None)
            optimal_abc = AdaBoostClassifier(base_estimator=DTC)
            optimal_abc.set_params(**self.hyper_para)
            optimal_abc.fit(self.X, self.y)
            y_pred = optimal_abc.predict(X)
            print(y_pred)
        # test model is Logistic Regression
        elif self.model_chosen == 'Logistic Regression':
            optimal_logreg = LogisticRegression()
            optimal_logreg.set_params(**self.hyper_para)
            optimal_logreg.fit(self.X, self.y)
            y_pred = optimal_logreg.predict(X)
            print(y_pred)
            numpy.savetxt('predicted-labels', y_pred, fmt='%d',delimiter=',')
        # test model is Naive Bayes, if pre-train method is word-embedding
        # Naive Bayes doesn't support continuous features
        # change the model or pre-train method
        elif self.model_chosen == 'Naive Bayes':
            if self.pre_train_chosen == 'word-embedding':
                print("not suitable for Naive Bayes classifier using word-embedding, because of the continuous feature")
            else:
                optimal_nb = MultinomialNB()
                optimal_nb.set_params(**self.hyper_para)
                optimal_nb.fit(self.X, self.y)
                y_pred = optimal_nb.predict(X)
                print(y_pred)


def main():
    # train datafile :'trainreviews.txt', test datafile:'testreviewsunlabeled.txt'
    # pre-train method:['bag-of-words','word-embedding']
    # model:['SVM', 'adaboost', 'Logistic Regression', 'Naive Bayes']
    # length of sentence: 50
    # train-test-ratio: 0.75
    classifier = MyClassifier(['trainreviews.txt', 'testreviewsunlabeled.txt', 'word-embedding', 'Logistic Regression', 50, 0.75])
    classifier.fit_choose_hyperparamter()
    classifier.test_evaluation_function()


if __name__ == "__main__":
    main()
