#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
#from keras.layers.recurrent import LSTM
lemmatizer = WordNetLemmatizer()
import json
import pickle


# In[60]:


w=[]

doc = []
c = []
L_ign = ['!', '?', ',', '.']
file = open('intents.json',encoding="utf8").read()

intents = json.loads(file)


# In[61]:


for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        word = nltk.word_tokenize(pattern) #tokenize word
        w.extend(word)
        
        doc.append((word, intent['tag']))
       
        if intent['tag'] not in c:
            c.append(intent['tag']) #Add a list to our classes
print(doc)


# In[62]:


# lemmaztize and lower each word and remove duplicates
w = [lemmatizer.lemmatize(w.lower()) for w in w if w not in L_ign] # lemmatize word 
w = sorted(list(set(w))) #remove duplicates word


# In[63]:



c = sorted(list(set(c)))
print (len(doc), "documents")
# classes = intents
print (len(c), "classes", c)
print (len(w), "unique lemmatized words", w) # all the uniq vaoc
pickle.dump(w,open('Vocabs.pkl','wb'))
pickle.dump(c,open('classes.pkl','wb'))


# In[64]:



train_data = []

output_empty = [0] * len(c)# create an empty array for our output

for i in doc: # training set, bag of words for each sentence
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = i[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in w:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[c.index(i[1])] = 1
    
    train_data.append([bag, output_row])

random.shuffle(train_data)
train_data  = np.array(train_data)


# In[65]:



train_x = list(train_data[:,0]) #Train_X : Patterns
train_y = list(train_data[:,1]) #Train_Y : Patterns
print("Training data created")


# In[66]:


# Create model - 3 layers. First layer 128 neurons, second layer 128 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output 
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]), activation='softmax'))


# In[67]:



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[68]:


#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('model.h5', hist)


# In[58]:


print("IT Chatbot Model")


# In[ ]:




