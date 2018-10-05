import pandas as pd

dataset_path_movies = '/movies.list.gz'
dataset_path_keywords = '/keywords.list.gz'
dataset_path_genres = '/genres.list.gz'
dataset_path_ratings = '/ratings.list.gz'


print("Opening Keyword File.")
cols_keywords = ['movie', 'keyword']
skip_keywords = 89713
df_keywords = pd.read_csv(dataset_path_keywords, header=None, names=cols_keywords, skiprows=skip_keywords, encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python')
print("Grouping Keywords.")
df_keywords = df_keywords.groupby('movie')['keyword'].apply(list).to_frame()


print("Opening Movie File.")
cols_movies = ['movie', 'years']
skip_movies = 15
skipfooter_movies = 1
df_movies = pd.read_csv(dataset_path_movies, header=None, names=cols_movies, skiprows=skip_movies, skipfooter=skipfooter_movies, encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python')


print("Opening Genres File.")
cols_genres = ['movie', 'genres']
skip_genres= 383
df_genres = pd.read_csv(dataset_path_genres, header=None, names=cols_genres, skiprows=skip_genres, encoding='ISO-8859-1', compression='gzip', sep='\t+', engine='python')
print("Grouping Genres.")
df_genres = df_genres.groupby('movie')['genres'].apply(list)


print("Opening Rating File.")
cols_ratings = ['distribution', 'votes', 'rank', 'title']
skip_ratings = 296
skipfooter_ratings = 146
df_ratings = pd.read_csv(dataset_path_ratings, header=None, names=cols_ratings, skiprows=skip_ratings, skipfooter=skipfooter_ratings, encoding='ISO-8859-1', compression='gzip', sep='\s{2,}', engine='python')
df_ratings = df_ratings.drop('distribution', 1)
