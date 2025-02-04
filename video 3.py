import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)
model = LightFM(loss = 'warp')

model.fit(data['train'], epochs = 30, num_threads = 2)

def sample_recommendations(model, data, user_ids):
    #number of users
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        print("run")
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        scores = model.predict(user_id, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]

        print(f"user {user_id}")
        print("     known positives:")

        print(known_positives)
        for x in known_positives:
            print(f"         {x}")


        print("     recommended:")

        for x in top_items[:3]:
            print(f"         {x}")


sample_recommendations(model, data, [3, 25, 450])