import numpy as np
import pandas as pd
import re
import scipy
import math

URL = 'https://drive.google.com/file/d/137eW4F35OctoRuq5DasVUXw6GpmfXdBS/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
#df = pd.read_pickle(path)
data = pd.read_csv(path, skip_blank_lines=True)
pd.set_option('display.max_colwidth', None)
print(data.shape)

data.drop_duplicates(subset='content', inplace=True, ignore_index=True)
data.shape

data.head(1)

data[data['_id'] == '6076fadb0b3e8bc9b779293e']['_id'].to_string()

def make_lower_case(text):
    return text.lower()

import re
from pprint import pprint
import nltk, spacy, gensim
from sklearn.feature_extraction.text import CountVectorizer

def get_lemmatized_clean_data(df):
    # Convert to list
    data = df.content.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    # pprint(data[:1])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # print(data_words[:1])

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized

X = get_lemmatized_clean_data(data)

max_time = []
for i in X:
  max_time.append(len(i.split(' '))/2.5)

data['Max_Time'] = max_time

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

def get_data(mydata):
    mydata.keys()
    df = pd.DataFrame([mydata['data'],[mydata['target_names'][idx] for idx in mydata['target']],mydata['target']])
    df = df.transpose()
    df.columns = ['content', 'target_names', 'target']
    return df

df = get_data(newsgroups_train)

df.head()

news = data.drop(axis = 1, columns=['_id',]).to_numpy()

data_lemmatized = get_lemmatized_clean_data(df)

df.head()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
my_stopwords = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizor = TfidfVectorizer(stop_words=my_stopwords, lowercase= True)
bag_of_words = vectorizor.fit_transform(X)

from numpy import dot
from numpy.linalg import norm

def similarity(a,b):
  cos_sim = dot(a, b)/(norm(a)*norm(b))
  return cos_sim

def ContentBasedFiltering(id, first_n = 10):
  similarity_dic = {}
  news_index = data[data['_id']==id].index[0]
  for i in data['_id']:
    an_index = data[data['_id']==i].index[0]
    a = np.array(bag_of_words[news_index].todense())[0]
    b = np.array(bag_of_words[an_index].todense())[0]
    similarity_dic[i] = similarity(a, b)

  sorted_most_similar = sorted(similarity_dic.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=True)

  return sorted_most_similar[:first_n]

ContentBasedFiltering('6076fadb0b3e8bc9b779293e')


for keys in ContentBasedFiltering('6076fadb0b3e8bc9b779293e'):
  print(data[data['_id'] == keys[0]]['title'])

# Performing SVD

svd = TruncatedSVD(n_components=50)
lsa = svd.fit_transform(bag_of_words)

def SVDContentBasedFiltering(id, first_n = 10):
  similarity_dic = {}
  news_index = data[data['_id']==id].index[0]
  for i in data['_id']:
    an_index = data[data['_id']==i].index[0]
    a = np.array(lsa[news_index])
    b = np.array(lsa[an_index])
    similarity_dic[i] = similarity(a, b)

  sorted_most_similar = sorted(similarity_dic.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=True)

  return sorted_most_similar[:first_n]

SVDContentBasedFiltering('6076fadb0b3e8bc9b779293e')

for keys in SVDContentBasedFiltering('6076fadb0b3e8bc9b779293e'):
  print(data[data['_id'] == keys[0]]['title'])

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
lda = LatentDirichletAllocation(learning_method='batch', n_jobs=-1)

bag_of_words.T

# LDA Cross-Validation
n_components = [20, 50, 70]
learning_decay = [0.5, 0.7, 0.8]


params = {'n_components': n_components, 'learning_decay': learning_decay}

model = GridSearchCV(lda, param_grid=params)
model.fit(bag_of_words.T)

best_params = model.best_estimator_
best_params

lda_res = best_params.components_.T

lda_res.shape

import pickle

pickle_file = 'lda_cross_validation_rev.pkl'

with open(pickle_file, 'wb') as file:
    pickle.dump(model, file)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# get the folder id where you want to save your file
file = drive.CreateFile({'parents':[{u'id': '19AI35wfuabh1JQ6b1Z3YH5uJ6uL3N6BD'}]})
file.SetContentFile(pickle_file)
file.Upload()

with open(pickle_file, 'rb') as file:
    lda_pkl_model = pickle.load(file)

def LDAContentBasedFiltering(id, first_n = 10):
  similarity_dic = {}
  news_index = data[data['_id']==id].index[0]
  for i in data['_id']:
    an_index = data[data['_id']==i].index[0]
    a = np.array(lda_res[news_index])
    b = np.array(lda_res[an_index])
    similarity_dic[i] = similarity(a, b)

  sorted_most_similar = sorted(similarity_dic.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=True)

  return sorted_most_similar[:first_n]

LDAContentBasedFiltering('6076fadb0b3e8bc9b779293e')

for keys in LDAContentBasedFiltering('6076fadb0b3e8bc9b779293e'):
  print(data[data['_id'] == keys[0]]['title'])

from google.colab import drive
drive.mount('/content/drive')

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('/content/drive/MyDrive/NLP and Text Analysis/glove.6B/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

i = 0
glov_stop = []
news_embedding_dict = dict()
for word in vectorizor.vocabulary_.keys():
  if word in embeddings_dictionary:
    news_embedding_dict[word] = embeddings_dictionary[word]
  else:
    glov_stop.append(word)


stopset = set(nltk.corpus.stopwords.words('english'))
new_stopwords_list = list(stopset.union(glov_stop))
vectorizor_glov = TfidfVectorizer(stop_words=new_stopwords_list)

glov_bag_of_words = vectorizor_glov.fit_transform(X)

y = np.array([val for (key, val) in news_embedding_dict.items()])

y.shape

glov_bag_of_words.shape

document_embedding = glov_bag_of_words*y

document_embedding.shape

document_embedding

def ContentBasedFilteringWordEmbedding(id, first_n = 10):
  similarity_dic = {}
  news_index = data[data['_id']==id].index[0]
  for i in data['_id']:
    an_index = data[data['_id']==i].index[0]
    a = np.array(document_embedding[news_index])
    b = np.array(document_embedding[an_index])
    similarity_dic[i] = similarity(a, b)

  sorted_most_similar = sorted(similarity_dic.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=True)

  return sorted_most_similar[:first_n]

ContentBasedFilteringWordEmbedding('6076fadb0b3e8bc9b779293e')

for keys in ContentBasedFilteringWordEmbedding('6076fadb0b3e8bc9b779293e'):
  print(data[data['_id'] == keys[0]]['title'])

topics = ['cricket', 'football', 'golf', 'asia', 'africa', 'europe', 'americas', 'style', 'tech', 'science', 'hollywood', 'us politics', 'stock market', 'travel', 'coronavirus', 'black lives matter']
from random import sample
class User:
  def __init__(self, id):
    self.id = id
    self.prefered_categories = sample(topics, np.random.randint(low=3, high= 5))
    self.no_of_articles_served = np.random.randint(10, 50)*10
    self.no_of_sessions = math.ceil((self.no_of_articles_served)/10)
    self.ids = [self.id for _ in range(self.no_of_articles_served)]
    self.sessions = []
    self.articles_served = []
    self.ratings = []
    self.click = []
    self.ranks = []
    j = math.ceil(self.no_of_articles_served*0.7)
    for m in range(j):
      id_temp = np.random.choice(data[data['topics'].isin(self.prefered_categories)]['_id'])
      self.articles_served.append(id_temp)
      click = np.random.binomial(1, 0.7,1)[0]
      self.click.append(click)
      self.ratings.append('-' if click == 0 else np.random.randint((data[data['_id'] == id_temp]['Max_Time'])/4, data[data['_id'] == self.articles_served[m]]['Max_Time'])[0])

    j = self.no_of_articles_served-j
    for m in range(j):
      id_temp = np.random.choice(data[~data['topics'].isin(self.prefered_categories)]['_id'])
      self.articles_served.append(id_temp)
      click = np.random.binomial(1, 0.1,1)[0]
      self.click.append(click)
      self.ratings.append('-' if click == 0 else np.random.randint(0, data[data['_id'] == id_temp]['Max_Time'])[0])
    for i in range(self.no_of_sessions):
      for k in range(10):
        self.sessions.append(i)
        self.ranks.append(k)


new_user = User(1)
data[data['_id'].isin(new_user.articles_served)].tail(10)

def CreateRandomUserProfiler(max_no_user = 40):
  Users = []
  for i in range(max_no_user):
    Users.append(User(i))
    print(Users[i-1].prefered_categories)


  UserProfiler = pd.DataFrame(columns=['UserId', 'SessionID', 'ArticleID Served', 'Article Rank', 'Click', 'Time Spent'])
  for user in Users:
    df = pd.DataFrame()
    df['UserId'] = user.ids
    df['SessionID'] = user.sessions
    df['ArticleID Served'] = user.articles_served
    df['Article Rank'] = user.ranks
    df['Click'] = user.click
    df['Time Spent'] = user.ratings
    UserProfiler = pd.concat([UserProfiler,df], ignore_index=True)

  return UserProfiler

UserProfiler = CreateRandomUserProfiler(40)

UserProfiler.head()

UserProfiler.shape

def getNewsInfo(id):
  return data[data['_id']==id]

import numpy as np
from scipy.sparse import csr_matrix

# Creating a user * news sparse matrix
sparseMatrix = csr_matrix((UserProfiler.UserId.unique().shape[0], data.shape[0])).toarray()

k = 0
user = UserProfiler.iloc[k]

for i in UserProfiler.UserId.unique():
  while user.UserId == i and k < UserProfiler.shape[0]:
    user = UserProfiler.iloc[k]
    if user.Click:
      newsInfo = getNewsInfo(user['ArticleID Served'])
      rating = user['Time Spent']/newsInfo['Max_Time']
      sparseMatrix[i][newsInfo.index] = rating
    k+=1

userItem = csr_matrix(sparseMatrix)
from numpy import count_nonzero
sparsity = 1.0 - count_nonzero(sparseMatrix) / sparseMatrix.size
print(sparsity)
pd.DataFrame(sparseMatrix)

def MF(X, num_dims, step_size,epochs,thres,lam_da):
  P = scipy.sparse.rand(X.shape[0], num_dims, 1, format='csr')
  P = scipy.sparse.csr_matrix(P/scipy.sparse.csr_matrix.sum(P, axis=1))
  Q = scipy.sparse.rand(num_dims, X.shape[1], 1, format='csr')
  Q = scipy.sparse.csr_matrix(Q/ scipy.sparse.csr_matrix.sum(Q, axis=0))

  prev_error = 0
  for iterat in range(epochs):
    errors = X - make_sparse(P.dot(Q).todense(), X)
    mse = np.sum(errors.multiply(errors))/len(X.indices)

    P_new=P + step_size*2*(errors.dot(Q.T)-lam_da*P)
    Q_new=Q + step_size*2*(P.T.dot(errors)-lam_da*Q)
    P=P_new
    Q=Q_new
    prev_error=mse
    #if iterat%1==0:
    print(iterat,mse)
  return pd.DataFrame(np.array(P.dot(Q).todense()))



def make_sparse(array, X):
  array = np.array(array)
  x_pos, y_pos = X.nonzero()
  # print(x_pos, y_pos)
  try2 = np.array([[0 for i in range(array.shape[1])] for j in range(array.shape[0])])
  l = len(x_pos)
  for i in range(l):
    # print(x_pos[i], y_pos[i])
    try2[x_pos[i]][y_pos[i]] = array[x_pos[i]][y_pos[i]]

  return scipy.sparse.csr_matrix(try2)

UserItem_MF = MF(userItem, 2, 0.005, 500, 0.00001, 5)

def GetRecommendations(UserID, top_n = 10):
  userRating = UserItem_MF[UserID]
  s = np.array(userRating)
  sort_index = np.argsort(s)
  print(sort_index)
  print('Last n topics',data.iloc[sort_index[-top_n:]]['topics'])
  print('First n topics',data.iloc[sort_index[:top_n]]['topics'])
  recommended_ids = data.iloc[sort_index[-top_n:]]['_id']
  return recommended_ids

ids = GetRecommendations(1)

for id in ids:
  print(getNewsInfo(id))

def getHybridRecommendations(user_id, last_read_article, top_n = 10):
  return GetRecommendations(user_id, math.ceil(top_n/2)).tolist() + [item[0] for item in SVDContentBasedFiltering(last_read_article, int(top_n/2))]


hybrid_ids = getHybridRecommendations(1, '6076faec0b3e8bc9b7792947')

for id in hybrid_ids:
  print(getNewsInfo(id))


