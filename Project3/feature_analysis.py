import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso



class MyOwnClassifier:
    def __init__(self, options):
        """

        :param options:
        """
        self.data = options[0]
        self.trainset = options[1]
        self.testset = options[2]
        self.gender = options[3]
        self.release_year = options[4]
        self.n_factors = 50
        self.reg_all = 0.04
        self.u = []
        self.v = []

    def model_fit(self):
        """
        train the SVD use the best parameter
        :return:
        """
        algo = SVD(n_factors=self.n_factors, reg_all=self.reg_all)
        print("best performance of SVD when using parameters n_factors:"
                + str(self.n_factors) + " and regularizer:" + str(self.reg_all))
        print()
        trainset = self.trainset.build_full_trainset()
        algo.fit(trainset)
        length = len(self.release_year)
        self.u = algo.pu
        self.v = algo.qi

    def result_anlysis_U(self):
        """
        use the U vector to predict the gender
        :return:
        """
        U = self.u
        y = self.gender.values
        tuned_parameters = [{'n_estimators': [10, 50, 100], 'max_depth':[1, 2]}]
        rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, refit=True)
        rf.fit(U, y.ravel())
        print('best parameters for random forest:')
        print(rf.best_params_)
        print('best training accuracy for random forest:')
        print(rf.best_score_)
        print()
        print('cross-validation results:')
        means = rf.cv_results_['mean_test_score']
        stds = rf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        optimised_rf = rf.best_estimator_



    def result_anlysis_V(self):
        """
        use V vector to predict the year
        :return:
        """
        V = self.v
        y = self.release_year.values
        tuned_parameters = [{'alpha': [1, 3, 5]}]
        make = make_scorer(mean_squared_error, greater_is_better=False)
        ridge = GridSearchCV(LinearRegression(), tuned_parameters, cv=5, refit=True, scoring=make)
        ridge.fit(V, y.ravel())
        print('best parameters for Ridge Regression:')
        print(ridge.best_params_)
        print()
        print('cross-validation results:')
        means = ridge.cv_results_['mean_test_score']
        for mean, params in zip(means, ridge.cv_results_['params']):
            print("%0.3f for %r" % (mean, params))
        optimised_ridge = ridge.best_estimator_
        optimised_ridge.fit(V, y.ravel())
        y_pred = optimised_ridge.predict(V)
        mse_ridge = mean_squared_error(y, y_pred)
        print('use best ridge regression model the MSE is:')
        print(mse_ridge)
        y_mean = np.mean(y)
        mean = np.full((len(y), 1), y_mean)
        mse_mean = mean_squared_error(y ,mean)
        print('use naive model the MSE is:')
        print(mse_mean)








def prepare_movielens_data(data_path):
    # get user gender, index user ids from 0 to (#user - 1)
    users = pd.read_csv(data_path + 'u.user', sep='|', header=None,
                        names=['id', 'age', 'gender', 'occupation', 'zip-code'])
    gender = pd.DataFrame(users['gender'].apply(lambda x: int(x == 'M')))  # convert F/M to 0/1
    user_id = dict(zip(users['id'], range(users.shape[0])))  # mapping user id to linear index

    # the zero-th column is the id, and the second column is the release date
    movies = pd.read_csv(data_path + 'u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1, 2],
                         names=['item-id', 'title', 'release-year'])

    bad_movie_ids = list(
        movies['item-id'].loc[movies['release-year'].isnull()])  # get movie ids with a bad release date

    movies = movies[movies['release-year'].notnull()]  # item 267 has a bad release year, remove this item
    release_year = pd.DataFrame(movies['release-year'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y').year))
    movie_id = dict(zip(movies['item-id'], range(movies.shape[0])))  # mapping movie id to linear index

    # get ratings, remove ratings of movies with bad release years.
    rating_triples = pd.read_csv(data_path + 'u.data', sep='\t', header=None,
                                 names=['user', 'item', 'rating', 'timestamp'])
    rating_triples = rating_triples[['user', 'item', 'rating']]  # drop the last column
    rating_triples = rating_triples[~ rating_triples['item'].isin(bad_movie_ids)]  # drop movies with bad release years

    # map user and item ids to user indices
    rating_triples['user'] = rating_triples['user'].map(user_id)
    rating_triples['item'] = rating_triples['item'].map(movie_id)

    # the following set assertions guarantees that the user ids are in [0, #users), and item ids are in [0, #items)
    assert (rating_triples['item'].unique().min() == 0)
    assert (rating_triples['item'].unique().max() == movies.shape[0] - 1)
    assert (rating_triples['user'].unique().min() == 0)
    assert (rating_triples['user'].unique().max() == users.shape[0] - 1)
    assert (rating_triples['item'].unique().shape[0] == movies.shape[0])
    assert (rating_triples['user'].unique().shape[0] == users.shape[0])

    # training/test set split
    rating_triples = rating_triples.sample(frac=1, random_state=2018).reset_index(drop=True)  # shuffle the data
    train_ratio = 0.9
    train_size = int(train_ratio * rating_triples.shape[0])

    trainset = rating_triples.loc[0:train_size]
    testset = rating_triples.loc[train_size + 1:]

    return trainset, testset, gender, release_year, rating_triples

def main():
    np.random.seed(0)
    print('Extracting data from the ml-100k dataset ...')
    # prepare dataset
    trainset, testset, gender, release_year, rating_triples = prepare_movielens_data(data_path='../ml-100k/')
    reader = Reader(sep='\t',rating_scale=(1, 5))
    data = Dataset.load_from_file('u.data', reader=reader)
    trainset_data = Dataset.load_from_df(rating_triples, reader=reader)
    print('setting parameters for the classifier...')
    classifier = MyOwnClassifier([data, trainset_data, testset, gender, release_year])
    print('fit the SVD model for cross-validation, please wait...')
    classifier.model_fit()
    classifier.result_anlysis_U()
    classifier.result_anlysis_V()
    print()
    print('Done')


if __name__ == '__main__':
    main()