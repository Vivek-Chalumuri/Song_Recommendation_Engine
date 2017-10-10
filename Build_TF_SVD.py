import pandas as pd
import tensorflow as tf
from sklearn.externals import joblib


song_df = pd.read_csv('user_song_count.csv')
#data_sample = song_df[:100000]
data_sample = song_df
users_sample = data_sample["user_id"].unique()
songs_sample = data_sample["song"].unique()

user_rank_matrix = data_sample.pivot_table(index='user_id', columns='song', values='listen_count', fill_value=0)

MAX_UID = user_rank_matrix.shape[0]
MAX_PID = user_rank_matrix.shape[1]

graph = tf.Graph()

K = 50
max_recs = 10

with graph.as_default():

    user_song_matrix = tf.placeholder(tf.float32, shape=(MAX_UID,MAX_PID))

    # SVD
    St, Ut, Vt = tf.svd(user_song_matrix)
    print("Vt: ",Vt.shape)

    # Compute reduced matrices
    Sk = tf.diag(St)[0:K, 0:K]
    Uk = Ut[:, 0:K]
    Vk = tf.transpose(Vt)[0:K, :]
    print("Vk: ",Vk.shape)
    print("Uk: ",Uk.shape)
    print("Sk: ",Sk.shape)

    # Compute Su and Si
    Su = tf.matmul(Uk, tf.sqrt(Sk))
    Si = tf.matmul(tf.sqrt(Sk), Vk)

    # Compute user average rating
    ratings_t = tf.matmul(Uk, Si)
    print(ratings_t.shape)

    # Pick top suggestions
    best_ratings_t, best_songs_t = tf.nn.top_k(ratings_t, max_recs)


session = tf.InteractiveSession(graph=graph)

feed_dict = {
    user_song_matrix: user_rank_matrix
}

best_songs = session.run([best_songs_t], feed_dict=feed_dict)

joblib.dump(best_songs, 'rec_sample.pkl')
joblib.dump(users_sample, 'users_sample.pkl')
joblib.dump(songs_sample, 'songs_sample.pkl')

print("_____DONE_____")

