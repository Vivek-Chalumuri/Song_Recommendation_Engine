import pandas
import numpy as np

class song_similarity_recommender():
    def __init__(self):
        self.data = None
        self.user_id = None
        self.song_name = None
        self.cooccurence_matrix = None
        
    def get_user_songs(self, user):
        user_data = self.data[self.data[self.user_id] == user]
        user_songs = list(user_data[self.song_name].unique())
        
        return user_songs
        
    def get_song_users(self, song):
        song_data = self.data[self.data[self.song_name] == song]
        song_users = set(song_data[self.user_id].unique())
            
        return song_users
        
    def get_all_songs_data(self):
        all_songs = list(self.data[self.song_name].unique())
            
        return all_songs
        
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_song_users(user_songs[i]))
            
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(0,len(all_songs)):
            
            songs_i_data = self.data[self.data[self.song_name] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                
                users_j = user_songs_users[j]
                    
                users_intersection = users_i.intersection(users_j)
                
                if len(users_intersection) != 0:
                    
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
        ######
        columns = ['user_id', 'song', 'score', 'rank']

        df = pandas.DataFrame(columns=columns)
         
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 5:
                ######
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the song similarity based recommendation model.")
            return -1
        else:
            return df

    def create_personal_song_recommender(self, data, user_id, song_name):
        self.data = data
        self.user_id = user_id
        self.song_name = song_name

    def recommend_songs(self, user):
        
        user_songs = self.get_user_songs(user)    
        ######

        all_songs = self.get_all_songs_data()
        ######
         
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations

class popularity_recommender():
    def __init__(self):
        self.data = None
        self.user_id = None
        self.song_id = None
        self.popularity_recommendations = None
        
    def create_popularity_recommender(self, data, user_id, song_id):
        self.data = data
        self.user_id = user_id
        self.song_id = song_id

        data_grouped = data.groupby([self.song_id]).agg({self.user_id: 'count'}).reset_index()
        data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        data_sort = data_grouped.sort_values(['score', self.song_id], ascending = [0,1])
    
        data_sort['Rank'] = data_sort['score'].rank(ascending=0, method='first')
        
        self.popularity_recommendations = data_sort.head(5)

    def recommend_songs(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        user_recommendations['user_id'] = user_id
    
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
