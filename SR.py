import pandas as pd
import Song_Recommender

Song_users = pd.read_csv('user_song_count.csv')
Song_users = Song_users.sort_values('user_id')
#Sample_data = Song_users[:100000]
Sample_data = Song_users
users_sample = Sample_data["user_id"].unique()
songs_sample = Sample_data["song"].unique()

def recommend(user_id):
    if user_id in users_sample:
        model = Song_Recommender.song_similarity_recommender()
        model.create_personal_song_recommender(Sample_data, 'user_id', 'song')
        print(model.recommend_songs(user_id))
    else:
        print("_________NEW USER__________")
        model = Song_Recommender.popularity_recommender()
        model.create_popularity_recommender(Sample_data, 'user_id', 'song')
        print(model.recommend_songs(user_id))


recommend(users_sample[0])
recommend('Mr. Nobody')

