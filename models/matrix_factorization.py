import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, SlopeOne, accuracy
import time
from loaders.data_loader import load_data
from surprise.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold

ratings, movies = load_data()

percentile = 0.05
user_rating_counts = ratings['userId'].value_counts()
num_top_users = int(np.ceil(len(user_rating_counts) * percentile))
top_users = user_rating_counts.nlargest(num_top_users).index.tolist()
ratings = ratings[ratings['userId'].isin(top_users)]

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

param_grid = {
    "n_factors": [x * 50 for x in range(1,5)],
    "n_epochs": [x * 10 for x in range(4, 7)],
    "lr_all": [x / 50 for x in range(1,6)],
    "reg_all": [0.01, 0.02, 0.03]

}
param_demo = {
    "n_factors": [x * 50 for x in range(1,2)],
    "n_epochs": [x * 10 for x in range(4,5)]
}
split = 5
gs = GridSearchCV(SVD, param_grid,
                   measures=['rmse', 'mae'], cv=split, 
                   n_jobs=6, refit=False)
gs.fit(data)
print('Best RMSE score: ', gs.best_score["rmse"])
print('Best param: ', gs.best_params["rmse"])

result = pd.DataFrame(gs.cv_results)
result.to_csv('./models/matrix_result.csv')

trainset, testset = train_test_split(data, test_size=0.2)
best_model = gs.best_estimator['rmse']
best_model.fit(trainset)

predictions = best_model.test(testset)

def round_to_half(x):
    return round(x * 2) / 2

rounded_predictions = [
    pred._replace(est=np.clip(round_to_half(pred.est), 0.5, 5.0))
    for pred in predictions
]
print(accuracy.mse(rounded_predictions))