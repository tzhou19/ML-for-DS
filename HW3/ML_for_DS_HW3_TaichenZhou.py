# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display_html
np.random.seed(42)

# %% [markdown]
# ### Problem1

# %% [markdown]
# ### a.

# %%
train = pd.read_csv('../HW3/hw3-data/ratings_train.csv', header=None)
train.columns = ['user_id', 'movie_id', 'rating']
test = pd.read_csv('../HW3/hw3-data/ratings_test.csv', header=None)
test.columns = ['user_id', 'movie_id', 'rating']
movie = pd.read_fwf('../HW3/hw3-data/movies.txt', header=None, sep='\n')
movie.columns = ['movie_name']
movie.head(10)
print(train.shape)
print(test.shape)
print(movie.shape)

# %%
SIGMA = 0.25
D = 10
LAMBDA = 1
N_1 = 943
N_2 = 1682
print("Number of users: ", N_1)
print("Number of movies: ", N_2)


# %%
class matrix_factorization:
    def __init__(self, n_1, n_2, d, n_iterations = 100) -> None:
        self.n_1 = n_1
        self.n_2 = n_2
        self.d = d
        self.n_iterations = n_iterations
        self.user = np.random.normal(0, LAMBDA, size = (n_1, self.d)) # n_1 x d
        self.movie = np.random.normal(0, LAMBDA, size = (n_2, self.d)) # n_2 x d


    def init_M(self, X):
        M = np.empty(shape = (self.n_1, self.n_2))
        M.fill(np.nan)
        for index, row in X.iterrows():
            user_id = int(row['user_id']) - 1
            movie_id = int(row['movie_id']) - 1
            M[user_id][movie_id] = row['rating']
        return M
        
    def fit(self, X):
        self.M = self.init_M(X)
        log_likelihood_list = []
        user_temp = self.user.copy()
        movie_temp = self.movie.copy()

        for iter in range(self.n_iterations):
            
            for u in range(self.n_1):
                omega_i = np.where(~np.isnan(self.M[u,:]))[0] # Index of movies rated by u
                if len(omega_i) == 0: # No moive have been rated by u
                    break
                # Number of movie rated by u -> k
                rated_movie = movie_temp[omega_i,:] # k x d
                rating_M = self.M[u, omega_i].reshape(1,-1) # 1 x k
                M_v = rating_M @ rated_movie # 1 x d
                # 1 x d     =  1 x d  @            d x d                      d x k         @   k x d
                self.user[u] = M_v @ np.linalg.inv((LAMBDA*SIGMA*np.eye(D)) + rated_movie.T @ rated_movie)
            user_temp = self.user.copy()

            for m in range(self.n_2):
                omega_i = np.where(~np.isnan(self.M[:,m]))[0] # Index of users rated m
                if len(omega_i) == 0: # No user have rated m
                    break
                # Number of user rated m -> k
                rated_user = user_temp[omega_i,:] # k x d
                rating_M = self.M[omega_i, m].reshape(1,-1) # 1 x k
                M_v = rating_M @ rated_user # 1 x d
                # 1 x d      =  1 x d @            d x d                       d x k        @   k x d
                self.movie[m] = M_v @ np.linalg.inv((LAMBDA*SIGMA*np.eye(D)) + rated_user.T @ rated_user)
            movie_temp = self.movie.copy()
            
            log_likelihood = self.log_likelihood()
            log_likelihood_list.append(log_likelihood)
        return log_likelihood_list

    def log_likelihood(self):
        # n1 x n2 = n1 x n2 - (n1 x d  @ d x n2)
        part1 = self.M - (self.user @ self.movie.T)
        part1 = part1[~np.isnan(part1)]
        loss = (-(1/SIGMA)*np.sum(part1**2) 
                - ((LAMBDA/2)*np.linalg.norm(self.user)**2) 
                - ((LAMBDA/2)*np.linalg.norm(self.movie)**2))
        return loss
    
    def predict(self, test):
        self.pred = np.zeros((test.shape[0]))
        for index, row in test.iterrows():
            user_id = int(row['user_id']) - 1
            movie_id = int(row['movie_id']) - 1
            self.pred[index] = self.user[user_id] @ self.movie[movie_id].T
        return self.pred

    def RMSE(self, test):
        pred = self.predict(test)
        rmse = np.sqrt(np.mean((pred - test['rating'].values)**2))
        return rmse
    

# %%
result = pd.DataFrame(columns=['Iteration', 'MAP', 'RMSE'])
log_likelihood_list = []
best_ll = -np.Infinity
best_model = None

for i in range(10):
    mf_model = matrix_factorization(N_1, N_2, D, 100)
    log_likelihood = mf_model.fit(train)
    pred = mf_model.predict(test)
    RMSE = mf_model.RMSE(test)
    log_likelihood_list.append(log_likelihood)
    result.loc[i] = [i, log_likelihood[-1], RMSE]
    if log_likelihood[-1] > best_ll:
        best_ll = log_likelihood[-1]
        best_model = mf_model

result.sort_values(by='MAP', ascending=False)
    
    

# %%
fig = plt.figure(figsize=(20,10))
for i in range(10):
    plt.plot(log_likelihood_list[i], label = 'Iteration {}'.format(i))
plt.legend()
plt.xlabel('Iteration')
plt.xlabel('Log likelihood')
plt.show()

# %%


# %% [markdown]
# ### b.

# %%
movie_temp = best_model.movie
user_temp = best_model.user

# %%
def get_top10(movie_name, movie_matrix, movie):
    movie_index = movie[movie['movie_name'].str.contains(movie_name)].index[0]
    moive_vector = movie_matrix[movie_index] # 1 x d
    movie_df = pd.DataFrame(columns=['moive_name', 'distance'])
    for i in range(movie_matrix.shape[0]):
        distance = np.linalg.norm(moive_vector - movie_matrix[i])
        movie_df.loc[i] = [movie.iloc[i][0], distance]

    movie_df.sort_values(by='distance', ascending=True, inplace=True)

    movie_df = movie_df.iloc[1:]

    return movie_df.head(10)

# %%
get_top10('Star Wars', movie_temp, movie)

# %%
get_top10('My Fair Lady', movie_temp, movie)

# %%
get_top10('GoodFellas', movie_temp, movie)

# %% [markdown]
# ### Problem2

# %% [markdown]
# ### a.

# %%
nyt_vocab = pd.read_csv('./hw3-data/nyt_vocab.dat', header=None)
nyt_vocab.shape

# %%
with open('./hw3-data/nyt_data.txt', 'r', encoding='utf-8-sig') as f:
    nyt_data = f.read().splitlines()

# %%
X_matrix = np.zeros((nyt_vocab.shape[0], len(nyt_data)))

for i in range(len(nyt_data)):
    for word in nyt_data[i].split(','):
        X_matrix[int(word.split(':')[0])-1, i] = int(word.split(':')[1])
X_matrix.shape

# %%
SIGMA = 0.25
D = 25
LAMBDA = 1
N_1 = 3012
N_2 = 8447
EPSILON = 1e-16
print("Number of words: ", N_1)
print("Number of documents: ", N_2)
print('Number of topics: ', D)

# %%
class NMF:
    def __init__(self, n_1, n_2, d, n_iterations = 100) -> None:
        self.n_1 = n_1
        self.n_2 = n_2
        self.d = d
        self.n_iterations = n_iterations
        self.word = np.random.uniform(1, 2, size = (n_1, self.d)) # n1 x d
        self.doc = np.random.uniform(1, 2, size = (n_2, self.d)) # n2 x d

    def fit(self, X_matrix):
        div_list = []
        for _ in range(self.n_iterations):
            doc_temp = self.doc
            word_temp = self.word
            #     n_1 x d   @  d x n_2
            self.WH = (self.word @ self.doc.T)
            # n2 x d =   n2 x d  *       n2 x n1    @ n1 x d    / 1 x d
            self.doc = self.doc * ((X_matrix/(self.WH+EPSILON)).T @ self.word)/(np.sum(self.word, 0).reshape(1,-1) + EPSILON)
            
            self.WH = (self.word @ self.doc.T)
            # n1 x d  = n1 x d  *       n1 x n2    @ n2 x d    / 1 x d
            self.word = self.word * (X_matrix/(self.WH+EPSILON) @ self.doc)/(np.sum(self.doc, 0).reshape(1,-1) + EPSILON)

            self.WH = (self.word @ self.doc.T)
            div = self.divergence(X_matrix)
            div_list.append(div)


        return div_list
    
    def divergence(self, X_matrix):
        return np.sum(X_matrix * np.log(1/(self.WH+EPSILON)) + self.WH)


# %%
NMF_model = NMF(N_1, N_2, D, 100)
div_list = NMF_model.fit(X_matrix)

# %%
fig = plt.figure(figsize=(20,10))
plt.plot(div_list)
plt.show()

# %% [markdown]
# ### b.

# %%
word_matrix = NMF_model.word
doc_matrix = NMF_model.doc
a = np.sum(word_matrix, 0).reshape(1,-1)
word_matrix = word_matrix/a
doc_matrix = doc_matrix/a


# %%
index_10 = np.argsort(word_matrix[:,0])[-10:]
nyt_vocab.iloc[index_10, 0]

# %%
np.sort(word_matrix[:,0])[-10:]

# %%
test_result = pd.DataFrame(columns = ['topic', 'word', 'weight'])
test_result['word'] = nyt_vocab.iloc[index_10, 0]
test_result['weight'] = np.sort(word_matrix[:,0])[-10:]
test_result['topic'] = 'topic{}'.format(0)
test_result

# %%
result = np.empty((5,5), dtype=object)
for i in range(5):
    for j in range(5):
        current_index = i*5+j
        result_temp = pd.DataFrame(columns = ['topic', 'word', 'weight'])
        index_10 = np.argsort(word_matrix[:,current_index])[-10:]
        result_temp['word'] = nyt_vocab.iloc[index_10, 0]
        result_temp['weight'] = np.sort(word_matrix[:,current_index])[-10:]
        result_temp['topic'] = 'topic{}'.format(current_index)
        result[i,j] = result_temp

# %%
for i in range(5):
    df1, df2, df3, df4, df5 = result[i][0], result[i][1], result[i][2], result[i][3], result[i][4]
    df1_styler = df1.style.set_table_attributes("style='display:inline'").set_caption('Topic{}'.format(i*5 + 0))
    df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption('Topic{}'.format(i*5 + 1))
    df3_styler = df3.style.set_table_attributes("style='display:inline'").set_caption('Topic{}'.format(i*5 + 2))
    df4_styler = df4.style.set_table_attributes("style='display:inline'").set_caption('Topic{}'.format(i*5 + 3))
    df5_styler = df5.style.set_table_attributes("style='display:inline'").set_caption('Topic{}'.format(i*5 + 4))
    display_html(df1_styler._repr_html_()
             +df2_styler._repr_html_()
             +df3_styler._repr_html_()
             +df4_styler._repr_html_()
             +df5_styler._repr_html_()
             , raw=True)

# %%



