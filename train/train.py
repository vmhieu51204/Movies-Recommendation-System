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
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

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

def round_to_half(x):
    return round(x * 2) / 2

def calculate_actor_average_ratings(df):
    movies_df = df.copy()
    actor_ratings = {}
    order_columns = ['order_0', 'order_1', 'order_2', 'order_3', 'order_4']
    
    print("Calculating actor average ratings...")
    
    for idx, row in movies_df.iterrows():
        movie_rating = row['avg_rating']
        if pd.isna(movie_rating):
            continue
            
        for col in order_columns:
            actor = row[col]
            if pd.isna(actor) or actor == '':
                continue
            if actor not in actor_ratings:
                actor_ratings[actor] = {'total_rating': 0, 'movie_count': 0}
            
            # Add rating and increment count
            actor_ratings[actor]['total_rating'] += movie_rating
            actor_ratings[actor]['movie_count'] += 1
    
    actor_avg_ratings = {}
    for actor, data in actor_ratings.items():
        actor_avg_ratings[actor] = data['total_rating'] / data['movie_count']
    
    all_actor_ratings = list(actor_avg_ratings.values())
    overall_avg_rating = np.mean(all_actor_ratings) if all_actor_ratings else 0
    
    print(f"Total unique actors: {len(actor_avg_ratings)}")
    print(f"Overall average rating across all actors: {overall_avg_rating:.3f}")
    
    print("Substituting actor average ratings in order columns...")
    
    for col in order_columns:
        movies_df[col] = movies_df[col].map(actor_avg_ratings)
        movies_df[col] = movies_df[col].fillna(overall_avg_rating)
    
    print("Substitution completed!")
    
    return movies_df, actor_avg_ratings, overall_avg_rating


def matrix_factorization(n_factors = 200, n_epochs = 60, lr_all = 0.02, reg_all = 0.02):
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    from surprise import Reader, Dataset, SVD, accuracy
    global ratings
    global movies
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    best_model = SVD(n_factors = 200, n_epochs = 60, lr_all = 0.02, reg_all = 0.02)
    trainset, testset = train_test_split(data, test_size=0.2)
    best_model.fit(trainset)
    predictions = best_model.test(testset)
    def round_to_half(x):
        return round(x * 2) / 2

    rounded_predictions = [
        pred._replace(est=np.clip(round_to_half(pred.est), 0.5, 5.0))
        for pred in predictions
    ]
    print("MSE: ", predictions)
    print("Rounded MSE: ",accuracy.mse(rounded_predictions))

def word_embed():
    global ratings
    global movies
    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=0.95, stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['soup'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                        columns=tfidf_vectorizer.get_feature_names_out(),
                        index=movies.index)

    print(f"Tf-idf matrix shape: ", tfidf_matrix.shape)

    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movies['id'])}
    user_movie_indices = [movie_id_to_index.get(movie_id) for movie_id in user_movies['id'].values]
    print("Number of ratings: ", len(user_ratings), ", Number of movies: ", len(user_movies))

    user_movie_indices = [idx for idx in user_movie_indices if idx is not None]

    for i in range(1,5):
        min_ratings = i * 100
        max_ratings = min_ratings + 50

        print(f"\nRating Interval: [{min_ratings}, {max_ratings})")
        
        # Select all users within the rating count interval
        user_counts = ratings['userId'].value_counts()
        eligible_users = user_counts[(user_counts >= min_ratings) & (user_counts < max_ratings)].index

        if len(eligible_users) == 0:
            print("No users in this rating interval. Skipping...")
            continue

        lr_mse_list = []
        lr_mae_list = []
        ridge_mse_list = []
        ridge_mae_list = []

        for user_id in eligible_users:
            user_ratings = ratings[ratings['userId'] == user_id]
            user_movies = user_ratings.merge(movies, left_on='movieId', right_on='id')

            user_movie_indices = [movie_id_to_index.get(movie_id) for movie_id in user_movies['id'].values]
            user_movie_indices = [idx for idx in user_movie_indices if idx is not None]

            if len(user_movie_indices) < 5:
                continue

            X = tfidf_matrix[user_movie_indices]
            y = user_movies.loc[user_movies['id'].isin(
                [movies['id'].iloc[idx] for idx in user_movie_indices]), 'rating'].values

            if len(y) < 5:
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Linear Regression
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_pred_raw = lr_model.predict(X_test)
                lr_pred = np.round(lr_pred_raw * 2) / 2
                lr_pred = np.clip(lr_pred, 0.5, 5.0)

                lr_mse = mean_squared_error(y_test, lr_pred)
                lr_mae = mean_absolute_error(y_test, lr_pred)
                lr_mse_list.append(lr_mse)
                lr_mae_list.append(lr_mae)

                # Ridge (SGDRegressor)
                sgd_model = SGDRegressor(penalty='l2', alpha=0.01, random_state=42)
                sgd_model.fit(X_train, y_train)
                ridge_pred_raw = sgd_model.predict(X_test)
                ridge_pred = np.round(ridge_pred_raw * 2) / 2
                ridge_pred = np.clip(ridge_pred, 0.5, 5.0)

                ridge_mse = mean_squared_error(y_test, ridge_pred)
                ridge_mae = mean_absolute_error(y_test, ridge_pred)
                ridge_mse_list.append(ridge_mse)
                ridge_mae_list.append(ridge_mae)

            except Exception as e:
                continue

        # Report averages for this interval
        if lr_mse_list:
            print("Average Model Evaluation:")
            print(f"Linear Regression - MSE: {np.mean(lr_mse_list):.4f}, MAE: {np.mean(lr_mae_list):.4f}")
            print(f"Ridge (SGDRegressor) - MSE: {np.mean(ridge_mse_list):.4f}, MAE: {np.mean(ridge_mae_list):.4f}")
        else:
            print("Not enough valid users/data to compute average metrics.")

def rating_genres():
    global movies
    global ratings
    modified_df, actor_ratings, overall_avg = calculate_actor_average_ratings(movies)
    mlb = MultiLabelBinarizer()
    genres_clean = modified_df['genres'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
    genres_encoded = mlb.fit_transform(genres_clean)
    genres_df = pd.DataFrame(genres_encoded, 
                            columns=[f'{genre}' for genre in mlb.classes_],
                            index=modified_df.index)
    modified_df = pd.concat([modified_df, genres_df], axis=1)
    
    feature_columns = [
    'order_0', 'order_1', 'order_2', 'order_3', 'order_4',
    'avg_rating', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
    'Thriller', 'War', 'Western'
    ]

    movies_features = modified_df[feature_columns].copy()
    movies_features.fillna(0, inplace=True) 

    scaler = StandardScaler()
    scaled_movie_features = scaler.fit_transform(movies_features)
    scaled_movie_features_df = pd.DataFrame(scaled_movie_features, columns=feature_columns, index=movies.index)


    print(f"Movie features shape: {scaled_movie_features_df.shape}")
    movie_id_to_index = {movie_id: i for i, movie_id in enumerate(movies['id'])}

    for i in range(1, 10):
        min_ratings = i * 50
        max_ratings = min_ratings + 50

        print(f"\nRating Interval: [{min_ratings}, {max_ratings})")

        # Select all users within the rating count interval
        user_counts = ratings['userId'].value_counts()
        eligible_users = user_counts[(user_counts >= min_ratings) & (user_counts < max_ratings)].index

        if len(eligible_users) == 0:
            print("No users in this rating interval. Skipping...")
            continue

        lr_mse_list = []
        lr_mae_list = []
        ridge_mse_list = []
        ridge_mae_list = []

        for user_id in eligible_users:
            user_ratings = ratings[ratings['userId'] == user_id]
            user_movies = user_ratings.merge(movies, left_on='movieId', right_on='id')

            user_movie_indices = [movie_id_to_index.get(movie_id) for movie_id in user_movies['id'].values]
            user_movie_indices = [idx for idx in user_movie_indices if idx is not None]

            if len(user_movie_indices) < 5: # Need at least 5 data points for train/test split
                continue

            X = scaled_movie_features_df.iloc[user_movie_indices].values
            y = user_movies.loc[user_movies['id'].isin(
                [movies['id'].iloc[idx] for idx in user_movie_indices]), 'rating'].values

            if len(y) < 5: 
                continue

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Linear Regression
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                lr_pred_raw = lr_model.predict(X_test)
                lr_pred = np.round(lr_pred_raw * 2) / 2
                lr_pred = np.clip(lr_pred, 0.5, 5.0)

                lr_mse = mean_squared_error(y_test, lr_pred)
                lr_mae = mean_absolute_error(y_test, lr_pred)
                lr_mse_list.append(lr_mse)
                lr_mae_list.append(lr_mae)

                # Ridge (SGDRegressor)
                sgd_model = SGDRegressor(penalty='l2', alpha=0.01, random_state=42, max_iter=1000) # Added max_iter
                sgd_model.fit(X_train, y_train)
                ridge_pred_raw = sgd_model.predict(X_test)
                ridge_pred = np.round(ridge_pred_raw * 2) / 2
                ridge_pred = np.clip(ridge_pred, 0.5, 5.0)

                ridge_mse = mean_squared_error(y_test, ridge_pred)
                ridge_mae = mean_absolute_error(y_test, ridge_pred)
                ridge_mse_list.append(ridge_mse)
                ridge_mae_list.append(ridge_mae)

            except Exception as e:
                continue
        if lr_mse_list:
            print("Average Model Evaluation:")
            print(f"Linear Regression - MSE: {np.mean(lr_mse_list):.4f}, MAE: {np.mean(lr_mae_list):.4f}")
            print(f"Ridge (SGDRegressor) - MSE: {np.mean(ridge_mse_list):.4f}, MAE: {np.mean(ridge_mae_list):.4f}")
        else:
            print("Not enough valid users/data to compute average metrics.")

def main(model_name, a = 0, b = 0, n_factors = 200, n_epochs = 10, lr_all = 0.005, reg_all = 0.02):
    if model_name == 'matrix':
        matrix_factorization(n_factors = n_factors, n_epochs = n_epochs, lr_all = lr_all, reg_all = reg_all)
    elif model_name == 'word_embed':
        word_embed()
    elif model_name == 'rating_genres':
        rating_genres()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a specified model with appropriate parameters.')
    parser.add_argument('model_name', type=str, help="Model to run: 'matrix', 'rating_genres', or 'word_embed'")

    parser.add_argument('--n_factors', type=int, default=200, help="Number of latent factors for matrix_factorization")
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs for matrix_factorization")
    parser.add_argument('--lr_all', type=float, default=0.005, help="Learning rate for matrix_factorization")
    parser.add_argument('--reg_all', type=float, default=0.02, help="Regularization term for matrix_factorization")

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all
    )
