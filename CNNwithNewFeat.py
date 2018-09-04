import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.layers.core import Dense, Dropout, Activation, Lambda
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.layers.convolutional import Conv1D
from keras import backend as K

df = pd.read_csv('featureSet.tsv',sep = '\t', header=0)


# In[3]:


df.head


# In[4]:


df.shape


# In[5]:


df['text'] = [str(x).lower() for x in df['text']]


# In[6]:


df['text']


# In[7]:


tokenizer = Tokenizer(num_words=4000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')


# In[8]:


tokenizer.fit_on_texts(df['text'])


# In[9]:


len(df['text'])


# In[10]:


train_Y = [str(x) for x in df['tag']]


# In[11]:


train_Y


# In[12]:


df['text'] = tokenizer.texts_to_sequences(df['text'])


# In[13]:


df.iloc[0][1:11]


# In[14]:


df['text'][0]


# In[15]:


new_x = []
new_text = []
new_y = []

# df = df.drop('UniCount', axis=1)
# df = df.drop('haveURL', axis=1)
# df = df.drop('totalPunct', axis=1)

# df = df.iloc[:, df.columns != 'SentLen' ]

for i in range(0, len(df['text'])):
    if len(df['text'][i]) < 400:
        new_text.append(df['text'][i])
        new_x.append(df.iloc[i][1:11])
        new_y.append(train_Y[i])


# In[16]:


print(len(new_x), len(new_x[0]))
# new


# In[17]:


df.iloc[0][0:]


# In[18]:


new_x = np.array(new_x)


# In[19]:


new_x


# In[20]:


len(new_y)


# In[21]:


text_pad = pad_sequences(new_text)


# In[22]:


text_pad.shape


# In[23]:


type(text_pad)


# In[24]:


type(new_x)


# In[25]:


X = np.concatenate((text_pad, new_x), axis=1)


# In[26]:


print(len(new_x), len(new_x[0]))
print(text_pad.shape)


# In[27]:


X.shape


# In[28]:


Y = pd.get_dummies(new_y)


# In[29]:


Y


# In[30]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)


# In[31]:


print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[32]:


# Our CNN Model

nb_filter = 100
kernel_size = 2
hidden_dims = 400
nb_epoch = 2

print('Build model...')
model = Sequential()
model.add(Embedding(4000, 128))
model.add(Dropout(0.2)) 
# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Conv1D(filters=nb_filter,
                        kernel_size=(kernel_size),
                        padding='valid',
                        activation='relu',
                        strides=1))

def max_1d(X_train):
    return K.max(X_train, axis=1)

model.add(Lambda(max_1d, output_shape=(nb_filter,)))

model.add(Dense(hidden_dims)) 
model.add(Dropout(0.2)) 
model.add(Activation('relu'))

model.add(Dense(hidden_dims)) 
model.add(Dropout(0.2)) 
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Acc: 0.7287722399230668


# In[33]:


model.summary()


# In[34]:


print('Train...')
model.fit(X_train, Y_train, batch_size=32, epochs=2,
          validation_split=0.2)
score, acc = model.evaluate(X_test, Y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)


# In[35]:


# Test score: 0.5160072511526885
# Test accuracy: 0.7287722399230668

# nb_filter = 100
# kernel_size = 2
# hidden_dims = 400
# epochs = 2


# In[36]:


predict = model.predict(X_test)


# In[37]:


predict


# In[ ]:





# In[38]:


(Y_test)


# In[39]:


test_y = Y_test.as_matrix()


# In[40]:


test_y


# In[41]:


type(predict)


# In[42]:


# p =precision_score(test_y, predict, average='weighted')


# In[43]:


len(predict)


# In[44]:


tmpMat = np.zeros((2324, 3), dtype=int)


# In[45]:


tmpMat


# In[46]:


for i in range(0,len(predict)):
    if(predict[i][0] > predict[i][1] and predict[i][0] > predict[i][2]):
        tmpMat[i][0]=1
    elif(predict[i][1] > predict[i][0] and predict[i][1] > predict[i][2]):
        tmpMat[i][1]=1
    else:
        tmpMat[i][2]=1
        


# In[47]:


tmpMat


# In[48]:


print(precision_score(test_y, tmpMat, average='weighted'))
print(recall_score(test_y, tmpMat, average='weighted'))
print(f1_score(test_y, tmpMat, average='weighted'))


print(precision_score(test_y, tmpMat, average='micro'))
print(recall_score(test_y, tmpMat, average='micro'))
print(f1_score(test_y, tmpMat, average='micro'))


print(precision_score(test_y[:,0], tmpMat[:,0], average='weighted'))
print(recall_score(test_y[:,0], tmpMat[:,0], average='weighted'))
print(f1_score(test_y[:,0], tmpMat[:,0], average='weighted'))



print(precision_score(test_y[:,1], tmpMat[:,1], average='weighted'))
print(recall_score(test_y[:,1], tmpMat[:,1], average='weighted'))
print(f1_score(test_y[:,1], tmpMat[:,1], average='weighted'))

print(precision_score(test_y[:,2], tmpMat[:,2], average='weighted'))
print(recall_score(test_y[:,2], tmpMat[:,2], average='weighted'))
print(f1_score(test_y[:,2], tmpMat[:,2], average='weighted'))


accuracy = (test_y == tmpMat).all(axis=(0,1)).mean()



print(accuracy)

error = np.mean( test_y != tmpMat)

print(error)

acc = np.mean(test_y == tmpMat)

print(acc)

