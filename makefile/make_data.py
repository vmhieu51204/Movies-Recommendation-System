import pandas as pd
import os
from ast import literal_eval
import argparse
from nltk.stem.snowball import SnowballStemmer

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

movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

columns_take = ['genres', 'id', 'title', 'description', 'cast', 'crew', 'keywords']
all_columns = movies.columns
columns_drop = [column for column in all_columns if column not in columns_take]
movies = movies.drop(columns=columns_drop)

movies['cast'] = movies['cast'].apply(literal_eval)
movies['crew'] = movies['crew'].apply(literal_eval)
movies['keywords'] = movies['keywords'].apply(literal_eval)
movies['cast_size'] = movies['cast'].apply(lambda x: len(x))
movies['crew_size'] = movies['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

movies['director'] = movies['crew'].apply(get_director)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies['cast'] = movies['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
movies['keywords'] = movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies['cast'] = movies['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
movies['director'] = movies['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
movies['director'] = movies['director'].apply(lambda x: [x,x, x])

s = movies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')

movies['keywords'] = movies['keywords'].apply(filter_keywords)
movies['keywords'] = movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
movies['soup'] = movies['keywords'] + movies['cast'] + movies['director'] + movies['genres']
movies['soup'] = movies['soup'].apply(lambda x: ' '.join(x))

movie_ids_in_ratings = ratings['movieId'].unique()
print(f"Number of movie IDs in ratings dataframe: {len(movie_ids_in_ratings)}")
movie_ids_in_movies = movies['id'].unique()
print(f"Number of movie IDs in movies dataframe: {len(movie_ids_in_movies)}")

missing_ids_ratings = [movie_id for movie_id in movie_ids_in_ratings 
                     if movie_id not in movie_ids_in_movies]
missing_ids_movies = [movie_id for movie_id in movie_ids_in_movies 
                     if movie_id not in movie_ids_in_ratings]

print(f"Number of movie IDs present in ratings but missing from movies: {len(missing_ids_ratings)}")
print(f"Percentage of missing movies: {len(missing_ids_ratings) / len(movie_ids_in_ratings) * 100:.2f}%")

print(f"Number of movie IDs present in movies but missing from ratings: {len(missing_ids_movies)}")
print(f"Percentage of missing movies: {len(missing_ids_movies) / len(movie_ids_in_movies) * 100:.2f}%")
# Ensure consistent types
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

valid_movie_ids = set(movies['id'])  # Convert to set for fast lookup

ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]
movies = movies[movies['id'].isin(ratings['movieId'].unique())]
print(f"New number of rows in ratings after removing invalid movie IDs: {len(ratings)}")

#print(f"New number of rows in ratings after removing invalid movie IDs: {len(ratings)}")
num_users = ratings['userId'].nunique()
num_items = ratings['movieId'].nunique()

user_mapping = {id: idx for idx, id in enumerate(ratings['userId'].unique())}
item_mapping = {id: idx for idx, id in enumerate(ratings['movieId'].unique())}
#convert non-sequential user IDs to sequential indices for matrix factorization
ratings['userId'] = ratings['userId'].map(user_mapping)
ratings['movieId'] = ratings['movieId'].map(item_mapping)
movies['id'] = movies['id'].map(item_mapping)

ratings.drop(columns=['timestamp'], inplace=True)
ratings.to_csv(os.path.join(path, 'p_ratings.csv'))
movies.to_csv(os.path.join(path, 'p_movies.csv'))