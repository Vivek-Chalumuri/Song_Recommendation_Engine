
import pandas as pd

count = 0

user_song_df = pd.DataFrame(columns=['song','user_id'])

for chunk in pd.read_csv('galvanize_songstart_sample.csv', chunksize=chunksize):
	df = pd.DataFrame({'user_id': chunk.UserId, 'song': chunk.Name})
	user_song_df  = user_song_df.append(df)
	count += 1
	print(count)
	print(user_song_df.shape)

user_song_df.to_csv('user_song.csv')

user_song_df = user_song_df.dropna(how='any')

user_song_df = user_song_df[user_song_df.song != '(null)']

user_song_df = user_song_df[user_song_df.user_id != '(null)']

user_song_df.to_csv('user_song.csv')

song_df = user_song_df.groupby(user_song_df.columns.tolist()).size().reset_index().rename(columns={0:'listen_count'})

song_df.to_csv('user_song_count.csv')

	