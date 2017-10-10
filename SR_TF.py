import pandas as pd
from sklearn.externals import joblib
import Song_Recommender

users_sample = list(joblib.load('users_sample.pkl'))
songs_sample = list(joblib.load('songs_sample.pkl'))
recommendation = joblib.load('rec_sample.pkl')

#print("User_ID: ",users_sample[0])

columns = ['User_id', 'Songs']

df = pd.DataFrame(columns=columns)

def recommend(user_id):
    if user_id in users_sample:
        index = users_sample.index(user_id)
        recs = recommendation[0][index][:5]
        df.Songs = [songs_sample[i] for i in recs]
        df.User_id = user_id
        print(df.head())
    else:
        Song_users = pd.read_csv('user_song_count.csv')
        Song_users = Song_users.sort_values('user_id')
        #Sample_data = Song_users[:100000]
        Sample_data = Song_users
        print("_________NEW USER__________")
        model = Song_Recommender.popularity_recommender()
        model.create_popularity_recommender(Sample_data, 'user_id', 'song')
        print(model.recommend_songs(user_id))

recommend(users_sample[3])
recommend('Mr. Apple')
recommend("000a8156-e072-47f0-8f92-828dcb860f54")
