import autograd.numpy as np
import pandas as pd
from autograd import grad
import matplotlib.pyplot as plt
import pickle

# Model
class MyLogisticReg:
    """A class for Linear Regression Models"""

    def __init__(self, options):
        """
        Put classifier options here as data attributes. Options include
        the type of gradients, regularizer weight, etc...
        Also put trainable classifier parameters here. The function
        fit below will learn these parameters
        """
        # option[0] = true means use all instances to calculate the w, w0.
        # option[0] = false means use SDG to calculate the w, w0.
        self.gradients_type = options[0]
        self.X_train = options[1]
        self.y_train = options[2]
        self.X_test = options[3]
        self.y_test = options[4]
        self.X = options[5]
        self.y = options[6]
        # initial w select zeros
        self.lamda = 0
        self.w_result = 0

    def lost(self, w):
          obj_reg = (1 / 2) * self.lamda * np.dot(w[1:], w[1:].T)
          if self.gradients_type == True:
                pred = np.add(np.dot(self.X_train, w[1:]),w[0])
                left = np.sum(np.multiply(self.y_train, pred))
                right_1 = -np.sum(np.log(1 + np.exp(pred[pred <= 30])))
                right_2 = -np.sum((pred[pred > 30]))
                obj_log =  left+right_1+right_2

          if self.gradients_type == False:
                data_idx = np.random.randint(0, len(self.X_train), 10)
                X_train = self.X_train[data_idx, :]
                y_train = self.y_train[data_idx]
                pred = np.add(np.dot(self.X_train, w[1:]),w[0])
                left = np.sum(np.multiply(self.y_train, pred))
                right_1 = -np.sum(np.log(1 + np.exp(pred[pred <= 30])))
                right_2 = -np.sum((pred[pred > 30]))
                obj_log =  left+right_1+right_2
                obj_log = obj_log * len(self.X_train) / 10

          loss = obj_reg - obj_log
          return loss



    def gradient(self, lost_func):
        return grad(lost_func)

    def fit(self, w):
        """
        Fit model, This function trains model parameters with input
        """
        w_former = w
        w_next = w
        w_t = w
        w_t_100 = w
        w_diff = 10000
        i = 0
        # use two part to calculate the a(w,w0):calculate the gradient using regular or SDG, batch = 10
        # calculate the gradient and update the w,w0
        while i < 10000 and np.abs(w_diff) > 0.00001:
            loss_func = self.lost
            grads = self.gradient(loss_func)
            # calculate the y_pred(eta)
            w_next = w_former - grads(w_former) / (10000)
            #k =self.lost(w_next) - self.lost(w_former)
            #m = np.dot(w_next-w_former, grads(w_former).T)
            if i != 0 and i % 100 == 0:
                w_t = w_t_100
                w_t_100 = w_next
                w_diff = 1 / len(w) * (np.sum(np.abs(w_t_100 - w_t)))
                i_loss = self.lost(w_next)
                print("Iteration < %d > with loss < %f >" % (i, i_loss))
            i += 1
            w_former = w_next
        if i >= 10000:
            print("~Optimization stops because finishing iteration~")
        if np.abs(w_diff) <= 0.00001:
            print("~Optimization stops because of difference between weights are less than 0.00001~")
        self.w_result = w_next

    def predict(self):
        """Predict using the logistic regression model"""
        add = np.ones(len(self.X_test))
        X_add = np.c_[add, self.X_test]
        pred = np.dot(X_add, self.w_result.T)

        pred[pred > 0] = 1
        pred[pred < 0] = 0
        return pred

    def evaluate(self, y_pred):

        accuracy = (np.sum(np.equal(self.y_test, y_pred).astype(np.float))
                      / self.y_test.size)
        return accuracy

        self.plt_accuracy_crossvalid(train_accuracy, test_accuracy, num_loops)
    def train_accuracy(self):
        """Evaluate the accuracy of predictions against true labels from train and test part"""
        # Train accuarcy
        add = np.ones(len(self.X_train))
        X_add1 = np.c_[add, self.X_train]
        pred_train = np.dot(X_add1, self.w_result.T)
        pred_train[pred_train > 0] = 1
        pred_train[pred_train < 0] = 0
        train_check_lable = np.isclose(pred_train, self.y_train)
        num_true_lable = np.sum(train_check_lable)
        num_all_lable = np.size(train_check_lable)
        train_accuracy = num_true_lable / num_all_lable
        print("train_accuracy is: %f" %train_accuracy)
        return train_accuracy

    def plt_accuracy(self, lamda_choice, weights_initial):
        """Draw accuracy corresponding to train and test data"""
        weights_random = []
        train_accuracy = []
        test_accuracy = []
        for i in range(len(lamda_choice)):
            self.lamda = lamda_choice[i]
            self.fit(weights_initial)
            pred_result = self.predict()
            test_ac = self.evaluate(pred_result)
            test_accuracy.append(test_ac)
            train_ac = self.train_accuracy()
            train_accuracy.append(train_ac)
            weights_random.append(self.w_result[-1])
        # print best lamda with highest accuracy
        print("choose lambda: %f" % lamda_choice[np.argmax(train_accuracy)])
        labels = ["Train_accuracy", "Test_accuracy"]
        fig, ax = plt.subplots()
        ax.plot(lamda_choice, train_accuracy, 'o-', label='Train_accuracy')
        ax.plot(lamda_choice, test_accuracy, 'o-', label='Test_accuracy')
        plt.xlabel('lambda choice')
        plt.ylabel('accuracy')
        # Draw absolute weight value corresponding the random feature
        fig, ax = plt.subplots()
        ax.plot(lamda_choice, weights_random, label='Weight for random')
        plt.xlabel('lambda_choice')
        plt.ylabel('weight_random')
        plt.show()
        # find the best lamda
        self.lamda = lamda_choice[np.argmax(train_accuracy)]

    def cross_valid_accuracy(self, X_trainfolder, y_trainfolder, X_testfolder,
                         y_testfolder, weight_initial):
        """calculate the train accuracy"""
        train_accuracy = []

        for i in range(len(X_trainfolder)):
            self.X_train = X_trainfolder[i]
            self.y_train = y_trainfolder[i]
            self.X_test = X_testfolder[i]
            self.y_test = y_testfolder[i]
            self.fit(weight_initial)
            train_accuracy.append(train_accuracy)
        print("cross_valid_accuracy mean = %f" % np.mean(train_accuracy))
        print("cross_valid_accuracy std = %f" % np.std(train_accuracy))


def split_data(data, train_test_ratio):
    """split the data"""
    data = data[data[:, 0].argsort()]
    number_one = np.count_nonzero(data[:, :1])
    number_zero = len(data) - number_one
    data_one = data[:number_one, :]
    data_zero = data[number_one:, :]
    batch_one_train = int(number_one * train_test_ratio / (1 + train_test_ratio))
    batch_zero_train = int(number_zero * train_test_ratio / (1 + train_test_ratio))
    train_sets = np.concatenate((data_one[:batch_one_train, :], data_zero[:batch_zero_train, :]), axis=0)
    test_sets = np.concatenate((data_one[batch_one_train:, :], data_zero[batch_zero_train:, :]), axis=0)
    np.random.shuffle(train_sets)
    np.random.shuffle(test_sets)
    return train_sets, test_sets


def split_data_crossvalid(data):
    """Split data using crossvalid"""
    X_trainfolder = []
    X_testfolder = []
    y_trainfolder = []
    y_testfolder = []
    data = data[data[:, 0].argsort()]
    number_one = np.count_nonzero(data[:, :1])
    data_one = data[np.where(data[:, 0] == 1)]
    data_zero = data[np.where(data[:, 0] == 0)]
    one_ratio = round(number_one / len(data), 1)
    one_zero_ratio = 1 - one_ratio
    batch_one = int(70 * one_ratio)
    batch_zero = int(70 * one_zero_ratio)
    batchs = len(data) // 70
    for i in range(batchs):
        test_one = data_one[i * batch_one:(i + 1) * batch_one, :]
        train_one = np.delete(data_one, test_one, axis = 0)
        test_zero = data_zero[i * batch_zero:(i + 1) * batch_zero, :]
        train_zero = np.delete(data_zero, test_zero, axis = 0)
        train_sets = np.concatenate((train_one, train_zero), axis=0)
        test_sets = np.concatenate((test_one, test_zero), axis=0)
        np.random.shuffle(train_sets)
        np.random.shuffle(test_sets)
        X_trainfolder.append(train_sets[:, 1:])
        y_trainfolder.append(train_sets[:, 0])
        X_testfolder.append(test_sets[:, 1:])
        y_testfolder.append(test_sets[:, 0])
    return X_trainfolder, y_trainfolder, X_testfolder, y_testfolder


def data_preparation(data, train_test_ratio, type_data_prep):
    # data preparation using regular
    if type_data_prep == True:
        train, test = split_data(data, train_test_ratio)
        min = np.min(train, axis=0, keepdims=True)  # take column min values
        max = np.max(train, axis=0, keepdims=True)
        train = (train - min) / (max - min)
        X_train = train[:, 1:]
        y_train = train[:, 0]
        X_test = test[:, 1:]
        y_test = test[:, 0]
        # data scaling
        X_train = X_train / 5
        X_test = X_test / 5
        return X_train, y_train, X_test, y_test
    # data preparation using cross-validation
    if type_data_prep == False:
        X_train, y_train, X_test, y_test = split_data_crossvalid(data)
        # data scaling
        #X_train = X_train / 5
        #X_test = X_test / 5
        return X_train, y_train, X_test, y_test

def main():
    data = pd.read_csv('titanic_train.csv')
    data = data.values 
    #divide data into feature and prediction
    X_train, y_train, X_test, y_test = data_preparation(data, 7/3, True)
    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]
    # model fitting
    model = MyLogisticReg([True, X_train, y_train, X_test, y_test, X, y])
    # Regularizer Selection
    weight_initial = np.random.rand(len(data[0]))
    #lamda_choice = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    #model.plt_accuracy(lamda_choice, weight_initial)
    #model.fit(weight_initial)
    #model.train_accuracy()
    #X_trainfolder, y_trainfolder, X_testfolder,y_testfolder = split_data_crossvalid(data)
    #model.cross_valid_accuracy(X_trainfolder, y_trainfolder, X_testfolder,y_testfolder, weight_initial)
    # prediction
    #y_pred = model.predict()
    # evaluation
    #test_accuracy = model.evaluate(y_pred)
    #print('The test accuracy of logistic regression is ' + str(test_accuracy))

    options = [True, model.X_train, model.y_train, model.X_test, model.y_test, X, y]
    test = MyLogisticReg(options)
    test.fit(weight_initial)
    pickle.dump(test,open("titanic_classifier.pkl","wb"))
    test_pickle = pickle.load(open("titanic_classifier.pkl","rb"))
    y_pred = test_pickle.predict()
    k = test.pickle.evaluate(y_pred)
if __name__=="__main__":
    main()
