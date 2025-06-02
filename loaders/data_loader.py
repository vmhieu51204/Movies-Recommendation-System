import os 
import pandas as pd
from config.config import read_config

config = read_config()

def load_data(path = config['path']):
    ratings = pd.read_csv(os.path.join(path, 'p_ratings.csv'))
    movies = pd.read_csv(os.path.join(path, 'p_movies.csv'))
    return ratings, movies

'''
ratings, movies = load_data(config['path'])
print(ratings.head())
print(movies.head())
'''