import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sklearn
import os
from gensim.models import Word2Vec
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from nltk.stem.porter import PorterStemmer
from tensorflow import keras
from keras.preprocessing import sequence
import matplotlib.pyplot as plt



#read bugreport for XLSX
projectname = 'Eclipse_Platform_UI_bugreport'
Eclipse_Platform_UI = pd.read_excel('dataset/' + projectname + '.xlsx', engine='openpyxl', nrows = 3400)#, nrows = 500
Eclipse_Platform_UI.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

#read Serverity for XLSX
serverity = pd.read_excel('dataset/' + 'serverity' + '.xlsx', engine='openpyxl', nrows = 3400)
serverity.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)
#concat serverity with bugreport

Eclipse_Platform_UI=pd.concat([Eclipse_Platform_UI, serverity],axis=1,ignore_index=True,keys='bug_id')

# Eclipse_Platform_UI.isnull().sum().plot.bar()
# plt.show()

#get bugid summary description
df_id_br_s=Eclipse_Platform_UI.iloc[:,[1,2,3,-1]]

# df_id_br_s.dropna(inplace = True)
# df_id_br_s.reset_index(inplace = True)
def merge_text(a,b):
    return (a,b)



X_me=df_id_br_s.apply(lambda row:merge_text(row[2],row[3]),axis=1).to_frame()

Y_me=df_id_br_s.iloc[:,-1].to_frame()

print(Y_me.describe())

#labale data y
# from keras.utils.np_utils import to_categorical
#
# Y_labels = to_categorical(Y_me)
from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()

Y_labels=OneHotEncoder(categories='auto').fit_transform(np.array(Y_me).reshape(-1,1)).toarray()
#_________________________________________________________________________________________________________
#read Serverity for XLSX    CommitSummary.csv

complexity = pd.read_csv('dataset/' + 'CommitSummary' + '.csv',nrows = 3400)
complexity.rename(columns = {'Unnamed: 10' : 'result'}, inplace = True)

complexity=complexity.iloc[:,[1,2,3,4,5]]



### Vocabulary size
voc_size=10000

messages = X_me.copy()
#messages.reset_index(inplace = True)
#messages['title']

#Download the nltk toolkit
# nltk.download('stopwords')
# nltk.download('punkt')


### Dataset Preprocessing
def process_data(messages):
    ps = PorterStemmer()
    corpus = []
    line=messages.shape[0]
    for i in range(0, messages.shape[0]):
        # temp=messages.loc[i,0]
        review = re.sub('[^a-zA-Z]', ' ', str(messages.loc[i,0]))
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

#split word
sentences= [nltk.word_tokenize(words) for words in process_data(messages)]

#word2vec
EMBEDDING_LEN=100
def get_word2vec_dictionaries(texts):

    Word2VecModel =Word2Vec(texts, window=5, min_count=3, workers=4,vector_size=EMBEDDING_LEN) #  Get the word2vector model
    words=list(Word2VecModel.wv.index_to_key)
    vocab_list = [word for word in words]  # Store all words  index_to_key enumerate(Word2VecModel.wv.index_to_key)


    word_index = {" ": 0}      # Initialize `[word: token]`, and later tokenize the corpus to use this dictionary.
    word_vector = {}           # Initialize the `[word: vector]` dictionary

    # Initialize , pay attention to one more bit (first row), the word vector is all 0, which is used for padding.
    # embeddings_matrix :The number of rows is the number of all words +1,
    # the number of columns is the "dimension" of the word vector, such as 100.
    embeddings_matrix = np.zeros((len(vocab_list) + 1, Word2VecModel.vector_size))

    ## Fill in the above dictionary and matrix
    for i in range(len(vocab_list)):
        word = vocab_list[i]  # Every word
        word_index[word] = i + 1  #Words: serial number
        word_vector[word] = Word2VecModel.wv[word] #Words: word vectors
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]  # Word vector matrix

    return word_index, word_vector, embeddings_matrix


word_index, word_vector, embeddings_matrix = get_word2vec_dictionaries(sentences)



MAX_SEQUENCE_LENGTH = 1500
# Serialize the text, tokenizer sentence, and return the word index corresponding to each sentence
def tokenizer(sentences, word_index):
    index_data = []
    for sentence in sentences:
        index_word = []
        for word in sentence:
            try:
                index_word.append(word_index[word])  # Convert the words to index
            except:
                index_word.append(0)
        index_data.append(index_word)

    #Use padding of kears to align sentences. The advantage is that the numpy array is output
    index_texts = sequence.pad_sequences(index_data, maxlen=MAX_SEQUENCE_LENGTH)
    return index_texts


X = tokenizer(sentences, word_index) #texts is numpy, input into the model calculation.

#embeddings_matrix.shape

# model = keras.Sequential([
#       keras.layers.Embedding(input_dim=embeddings_matrix.shape[0],
#                              output_dim=embeddings_matrix.shape[1],
#                              weights=[embeddings_matrix],
#                              input_length=1500),
#       keras.layers.LSTM(200),
#       keras.layers.LSTM(64),
#       keras.layers.Dropout(0.3),
#       keras.layers.Dense(7, activation='softmax')
#    ])

#________________________________________________________________
mix_max_scaler=sklearn.preprocessing.MinMaxScaler()
complexity_scaler=mix_max_scaler.fit_transform(complexity)
#________________________________________________________________
bugreport_input = keras.Input(shape=(X.shape[1],),name="bugreport")
complexity_input = keras.Input(shape=(complexity_scaler.shape[1],),name="complexity")
features =Embedding(output_dim=embeddings_matrix.shape[1],
                         input_dim=embeddings_matrix.shape[0],
                         weights=[embeddings_matrix],
                         input_length=1500)(bugreport_input)
lstm_out = LSTM(128)(features)
hidden_x = Dense(64, activation='tanh')(lstm_out)
concatenate_layer=tf.keras.layers.concatenate([hidden_x,complexity_input], axis=1)
hidden_x = Dense(64, activation='tanh')(concatenate_layer)
output = Dense(7, activation='softmax')(hidden_x)

model = keras.Model(inputs=[bugreport_input,complexity_input],outputs=output)
#________________________________________________________________

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'],)

model.summary()


X_final=X  #np.array(X)
y_final=np.array(Y_labels)

# X_final.shape,y_final.shape

mix_max_scaler=sklearn.preprocessing.MinMaxScaler()
X_final=mix_max_scaler.fit_transform(X_final)

# X_final = sklearn.preprocessing.scale(X_final)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.30, random_state=42)

length=len(X_train)


complexity_train=complexity_scaler[:length,:]
complexity_test=complexity_scaler[length:,:]
### Finally Training
# model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=100)

model.fit([X_train,complexity_train],y_train,validation_data=([X_test,complexity_test],y_test),epochs=20)


# results = model.evaluate(X_test, y_test)
# print(results)
# predictions = model.predict(X_test)
# print(predictions)

results = model.evaluate(X_test,complexity_test,y_test)
print('evaluate test data:')
print(results)




