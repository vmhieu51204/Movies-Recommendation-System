import numpy as np 
import pandas as pd 
import os
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="enter the path to the folder with the dataset csv files",
                    type=str)
parser.add_argument("model_path", help="enter the path to save the trained model",
                    type=str)
parser.add_argument("model", help="select: matrix, linear, ridge"
                    type=str)
parser.add_argument('--a', type=int, help="Value for a")
parser.add_argument('--b', type=int, help="Value for b")

args = parser.parse_args()
path = args.path
model_path = args.model_path
model_name = args.model
ratings = pd.read_csv(os.path.join(path, 'p_ratings.csv'))
movies = pd.read_csv(os.path.join(path, 'p_movies_metadata.csv'))

tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.95, stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['description'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                       columns=tfidf_vectorizer.get_feature_names_out(),
                       index=movies.index)

print(f"Tf-idf matrix shape: ", tfidf_matrix.shape)

def select_user(a, b):
    user_ratings_count = ratings.groupby('userId').size()
    eligible_users = user_ratings_count[(user_ratings_count >= a) & (user_ratings_count <= b)].index.tolist()
    if not eligible_users:
        raise ValueError(f"No users found with ratings count in range [{a}, {b}]")
    #selected_user = eligible_users[np.random.randint(0, len(eligible_users))]
    selected_user = eligible_users[0]
    print(f"Selected user {selected_user} with {user_ratings_count[selected_user]} ratings")

    user_ratings = ratings[ratings['userId'] == selected_user]
    user_movies = user_ratings.merge(movies, left_on='movieId', right_on='id')

    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movies['id'])}
    user_movie_indices = [movie_id_to_index.get(movie_id) for movie_id in user_movies['id'].values]
    print("Number of ratings: ", len(user_ratings), ", Number of movies: ", len(user_movies))

    # Filter out any None values (movies not in the mapping)
    user_movie_indices = [idx for idx in user_movie_indices if idx is not None]

    # Extract features using these indices
    X = tfidf_matrix[user_movie_indices]
    y = user_movies.loc[user_movies['id'].isin([movies['id'].iloc[idx] for 
                                                idx in user_movie_indices]), 'rating'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return selected_user, X_train, X_test, y_train, y_test


def round_to_half(x):
    return round(x * 2) / 2

def matrix_factorization(n_factors = 200, n_epochs = 10, lr_all = 0.005, reg_all = 0.02):
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    from surprise import Reader, Dataset, SVD, accuracy
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD(n_factors = n_factors, n_epochs = n_epochs, lr_all = lr_all, reg_all = reg_all)
    model.fit(trainset)
    predictions = model.test(testset)
    rounded_predictions = [
        pred._replace(est=np.clip(round_to_half(pred.est), 0.5, 5.0))
        for pred in predictions
    ]
    rmse = accuracy.rmse(predictions)
    mse = accuracy.mse(predictions)
    mae = accuracy.mae(rounded_predictions)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    with open(os.path.join(model_path, 'matrix_factorization_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

def linear(a,b):
    selected_user, X_train, X_test, y_train, y_test = select_user(a,b)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred_raw = lr_model.predict(X_test)

    lr_pred = np.round(lr_pred_raw * 2) / 2
    lr_pred = np.clip(lr_pred, 0.5, 5.0)

    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = root_mean_squared_error(y_test, lr_pred)
    print("User: ", selected_user)
    print(f"Test RMSE: {lr_rmse:.4f}")
    print(f"Test MSE: {lr_mse:.4f}")
    print(f"Test MAE: {lr_mae:.4f}")

    with open(os.path.join(model_path, 'linear_model.pkl'), 'wb') as f:
        pickle.dump(lr_model, f)

def ridge(a,b):
    selected_user, X_train, X_test, y_train, y_test = select_user(a,b)
    sgd_model = SGDRegressor(penalty='l2', alpha=0.01, random_state=42)
    sgd_model.fit(X_train, y_train)

    ridge_pred_raw = sgd_model.predict(X_test)
    ridge_pred = np.round(ridge_pred_raw * 2) / 2
    ridge_pred = np.clip(ridge_pred, 0.5, 5.0)

    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = root_mean_squared_error(y_test, ridge_pred)
    print("User: ", selected_user)
    print(f"Test RMSE: {ridge_rmse:.4f}")
    print(f"Test MSE: {ridge_mse:.4f}")
    print(f"Test MAE: {ridge_mae:.4f}")
    
    with open(os.path.join(model_path, 'ridge_model.pkl'), 'wb') as f:
        pickle.dump(sgd_model, f)

def main(model_name, a = 0, b = 0, n_factors = 200, n_epochs = 10, lr_all = 0.005, reg_all = 0.02):
    if model_name == 'matrix':
        matrix_factorization(n_factors = n_factors, n_epochs = n_epochs, lr_all = lr_all, reg_all = reg_all)
    elif model_name == 'linear':
        linear(a,b)
    elif model_name == 'ridge':
        ridge(a,b)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a specified model with appropriate parameters.')
    parser.add_argument('model_name', type=str, help="Model to run: 'matrix', 'linear', or 'ridge'")

    parser.add_argument('--a', type=int, help="Parameter a (used in linear and ridge)")
    parser.add_argument('--b', type=int, help="Parameter b (used in linear and ridge)")

    parser.add_argument('--n_factors', type=int, default=200, help="Number of latent factors for matrix_factorization")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs for matrix_factorization")
    parser.add_argument('--lr_all', type=float, default=0.005, help="Learning rate for matrix_factorization")
    parser.add_argument('--reg_all', type=float, default=0.02, help="Regularization term for matrix_factorization")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        a=args.a,
        b=args.b,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all
    )
