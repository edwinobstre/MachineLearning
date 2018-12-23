import numpy as np
import pandas as pd
from datetime import datetime
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from collections import defaultdict


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
        self.n = 5
        self.n_factors = 0
        self.reg_all = 0

    def param_selection(self):
        """
        select the best parameter for SVD, using cross-validation
        :param data:
        :return: SVD paramters
        """
        tuned_parameters = {'n_factors': [20, 50, 100], 'reg_all': [0.04, 0.05]}
        grid_search = GridSearchCV(SVD, tuned_parameters, measures=['rmse', 'mae'], cv=3)
        grid_search.fit(self.trainset)
        print("Best parameters using RMSE:")
        print(grid_search.best_params['rmse'])
        print()
        self.n_factors = grid_search.best_params['mae'].get('n_factors')
        self.reg_all = grid_search.best_params['mae'].get('reg_all')
        print("Best score using RMSE:")
        print(grid_search.best_score['rmse'])
        print()
        print("Best parameters using MAE:")
        print(grid_search.best_params['mae'])
        print()
        print("Best score using MAE:")
        print(grid_search.best_score['mae'])
        print()

    def model_fit(self):
        """
        refit the best SVD model
        :return: prediction
        """

        algo = SVD(n_factors=self.n_factors, reg_all=self.reg_all)
        trainset = self.trainset.build_full_trainset()
        print("best performance of SVD when using parameters n_factors:"
              + str(self.n_factors) + " and regularizer:" + str(self.reg_all))
        print()
        print('refit the model to make prediction, please wait...')
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        return predictions

    def model_mae(self):
        """
        calculate the MAE
        :return:
        """

        reader = Reader(sep='\t', rating_scale=(1, 5))
        testset = Dataset.load_from_df(self.testset, reader=reader)
        algo = SVD(n_factors=self.n_factors, reg_all=self.reg_all)
        trainset = self.trainset.build_full_trainset()
        algo.fit(trainset)
        testset = testset.build_full_trainset()
        testset = testset.build_testset()
        predictions = algo.test(testset)
        print("MAE for the prediction")
        mae = accuracy.mae(predictions)
        print()


    def model_test(self):
        """
        get the average rating for the top 5 recommendation movies
        :return:
        """
        top_n = self.get_top_n(self.model_fit())
        sum = 0
        temp = 0
        total = 0
        test = self.put_testset_in_dict()
        with open('./r.txt', 'wt') as f:
            print('user id , average rating', file=f)
            for uid, user_ratings in top_n.items():
                iid = [i[0] for i in user_ratings]
                for j in range(5):
                    if (int(uid), int(iid[j])) in test:
                        sum += test.get((int(uid), int(iid[j])))
                    else:
                        temp = 2
                    sum += temp
                    temp = 0
                total += sum
                average = sum/5
                sum = 0
                print(uid, ',', average, file=f)
            print('total average rating:', total/(943*5))

    def get_top_n(self, predictions):
        """
        get top n recommendations for each user
        :param predictions:
        :return:
        """
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:self.n]

        return top_n

    def put_testset_in_dict(self):
        """
        put testset in a dictionary, key is (user,item), value is rating
        :return:
        """
        testset_data = self.testset.values
        testset = defaultdict()
        for i in range(len(testset_data)):
            testset[(testset_data[i][0], testset_data[i][1])] = testset_data[i][2]
        return testset


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

    return trainset, testset, gender, release_year


def main():
    np.random.seed(0)
    print('Extracting data from the ml-100k dataset ...')
    # prepare dataset
    trainset, testset, gender, release_year = prepare_movielens_data(data_path='../ml-100k/')
    reader = Reader(sep='\t',rating_scale=(1, 5))
    data = Dataset.load_from_file('u.data', reader=reader)
    trainset_data = Dataset.load_from_df(trainset, reader=reader)
    print('setting parameters for the classifier...')
    classifier = MyOwnClassifier([data, trainset_data, testset, gender, release_year])
    print('fit the SVD model for cross-validation, please wait...')
    classifier.param_selection()
    classifier.model_mae()
    classifier.model_fit()
    classifier.model_test()
    print()
    print('Done')


if __name__ == '__main__':
    main()
