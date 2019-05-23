#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import pickle

# open a file, where you stored the pickled data
file = open('data.pickle', 'rb')

# dump information to that file
dataset,datalabels = pickle.load(file)

# close the file
file.close()

train_set = dataset[0:4000]
train_labels = datalabels[0:4000]
test_set = dataset[4000:]
test_labels = datalabels[4000:]


model = Sequential()
model.add(Dense(2, input_dim=5282, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(np.array(train_set), train_labels, epochs=30, batch_size=2, verbose=2)
model.save('my_model.h5') 


# In[19]:



predicted = model.predict_classes(np.array(test_set))

count = 0
  
for i in range(len(predicted)):
    if predicted[i] == test_labels[i]:
        count+=1
        
print('accuracy {} %'.format((count*100.0)/len(predicted)))


# In[ ]:




