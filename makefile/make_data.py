import pandas as pd
import os
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="enter the path to the folder with the dataset csv files",
                    type=str)
args = parser.parse_args()
path = args.path

credits = pd.read_csv(os.path.join(path, 'credits.csv'))
keywords = pd.read_csv(os.path.join(path, 'keywords.csv'))
links_small = pd.read_csv(os.path.join(path, 'links_small.csv'))
ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))
movies = pd.read_csv(os.path.join(path, 'movies_metadata.csv'))

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
movies = movies.drop([19730, 29503, 35587])
movies['id'] = movies['id'].astype('int')
movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies['description'] = movies['overview']
movies['description'] = movies['description'].fillna('')
#movies = movies[movies['id'].isin(links_small)]

columns_take = ['genres', 'id', 'title', 'description']
all_columns = movies.columns
columns_drop = [column for column in all_columns if column not in columns_take]
movies = movies.drop(columns=columns_drop)

movies.to_csv(os.path.join(path, 'p_movies.csv'))

movie_ids_in_ratings = ratings['movieId'].unique()
#print(f"Number of movie IDs in ratings dataframe: {len(movie_ids_in_ratings)}")
movie_ids_in_movies = movies['id'].unique()
#print(f"Number of movie IDs in movies dataframe: {len(movie_ids_in_movies)}")

missing_ids_ratings = [movie_id for movie_id in movie_ids_in_ratings 
                     if movie_id not in movie_ids_in_movies]
missing_ids_movies = [movie_id for movie_id in movie_ids_in_movies 
                     if movie_id not in movie_ids_in_ratings]

#print(f"Number of movie IDs present in ratings but missing from movies: {len(missing_ids_ratings)}")
#print(f"Percentage of missing movies: {len(missing_ids_ratings) / len(movie_ids_in_ratings) * 100:.2f}%")

#print(f"Number of movie IDs present in movies but missing from ratings: {len(missing_ids_movies)}")
#print(f"Percentage of missing movies: {len(missing_ids_movies) / len(movie_ids_in_movies) * 100:.2f}%")

# Ensure consistent types
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

valid_movie_ids = set(movies['id'])  # Convert to set for fast lookup
ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]

#print(f"New number of rows in ratings after removing invalid movie IDs: {len(ratings)}")

ratings.drop(columns=['timestamp'], inplace=True)
ratings.to_csv(os.path.join(path, 'p_ratings.csv'))
