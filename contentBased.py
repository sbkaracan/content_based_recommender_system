#############################
# Content Based Recommendation System
#############################



#################################
# Creation of TF-IDF Matrix
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)
df.head()

df["overview"].head()

df = df[:20000]

from sklearn.feature_extraction.text import TfidfVectorizer



#################################
# TF-IDF
#################################

df['overview'].head()

tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
tfidf_matrix.shape
# (45466, 75827)


#################################
# 2. Creation of Cosine Similarity Matrix
#################################


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
# (20000, 20000)

cosine_sim[1]
"""
array([0.01575748, 1.        , 0.04907345, ..., 0.        , 0.        ,
       0.        ])
"""

#################################
# 3. Making Recommendations Based on Similarities
#################################


indices = pd.Series(df.index, index=df['title'])
indices = indices[~indices.index.duplicated(keep='last')]


indices[:10]
"""
title
Toy Story                       0
Jumanji                         1
Grumpier Old Men                2
Waiting to Exhale               3
Father of the Bride Part II     4
Tom and Huck                    7
Sudden Death                    8
GoldenEye                       9
The American President         10
Dracula: Dead and Loving It    11
dtype: int64
"""


indices["Sherlock Holmes"]
# 17830

movie_index = indices["Sherlock Holmes"]


cosine_sim[movie_index]
# array([0., 0., 0., ..., 0., 0., 0.])

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]
"""
3166                    They Might Be Giants
4434                          Without a Clue
14557                        Sherlock Holmes
2301                   Young Sherlock Holmes
9743             The Seven-Per-Cent Solution
6432     The Private Life of Sherlock Holmes
5928                        Murder by Decree
14827    The Case of the Whitechapel Vampire
18258     Sherlock Holmes: A Game of Shadows
5249           The Hound of the Baskervilles
Name: title, dtype: object
"""

#################################
# 4. Script of the code
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)
"""
3166                    They Might Be Giants
4434                          Without a Clue
14557                        Sherlock Holmes
2301                   Young Sherlock Holmes
9743             The Seven-Per-Cent Solution
6432     The Private Life of Sherlock Holmes
5928                        Murder by Decree
14827    The Case of the Whitechapel Vampire
18258     Sherlock Holmes: A Game of Shadows
5249           The Hound of the Baskervilles
Name: title, dtype: object
"""


content_based_recommender("The Matrix", cosine_sim, df)
"""
167                       Hackers
6515                     Commando
9159                     Takedown
10815                       Pulse
9372                The Animatrix
10948                          23
7897             Freedom Downtime
13887        The Inhabited Island
14551                      Avatar
12818    War Games: The Dead Code
Name: title, dtype: object
"""

content_based_recommender("The Godfather", cosine_sim, df)
"""
1178      The Godfather: Part II
1914     The Godfather: Part III
11297           Household Saints
10821                   Election
17729          Short Sharp Shock
8653                Violent City
13177               I Am the Law
6711                    Mobsters
6977             Queen of Hearts
18224                  Miss Bala
Name: title, dtype: object
"""

content_based_recommender('The Dark Knight Rises', cosine_sim, df)
"""
12481                            The Dark Knight
150                               Batman Forever
1328                              Batman Returns
15511                 Batman: Under the Red Hood
585                                       Batman
9230          Batman Beyond: Return of the Joker
18035                           Batman: Year One
19792    Batman: The Dark Knight Returns, Part 1
3095                Batman: Mask of the Phantasm
10122                              Batman Begins
Name: title, dtype: object
"""
