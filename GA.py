#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random
import numpy as np


# In[36]:


'''
Create a model without assigning the Model Weight, and the Model has only 1 hidden layer with twice as many nodes as the input dimensionality
'''
def create_model_null():
    model = Sequential()
    model.add(Dense(68,input_dim = 34, activation = 'sigmoid'))
    model.add(Dense(1,activation = 'sigmoid'))
    return model


# In[37]:


'''
Create a model with specific Model Weight, and the Model has only 1 hidden layer with twice as many nodes as the input dimensionality
'''
def create_model(weight1,bias1,weight2,bias2):
    L1 = []
    L1.append(weight1)
    L1.append(bias1)
    L1.append(weight2)
    L1.append(bias2)
    np.asarray(L1)
    model = Sequential()
    model.set_weights(L1)
    model.add(Dense(68,input_dim = 34, activation = 'sigmoid'))
    model.add(Dense(1,activation = 'sigmoid'))
    return model


# In[38]:


'''
Self created algorithm for generating a new array by crossing over two arrays
'''
def swap(arr1,arr2):
    for index in range(1,len(arr1),2):
        arr1[index] = arr2[index]
    '''
    for i in range(0,len(arr1)):
        for j in range(0,len(arr1[i]),1):
            arr1[i][j] = arr2[i][j]
    '''
    return arr1


# In[39]:


def create_Initial():
    weight1 = np.random.rand(34,68)
    bias1 = np.random.rand(1,68)
    weight2 = np.random.rand(68,1)
    bias2 = np.random.rand(1,1)
    return weight1,bias1,weight2,bias2


# In[40]:


'''
Generate new weight for next generation's model  ----- This homework, i created 10 models per generation
'''
def generate_new_weight(weight1,bias1,weight2,bias2,scores):
    Arr = []
    scores_copy = []
    scores_copy = sorted(scores,reverse = True)
    for elem in scores_copy:
        for index in range(0,len(scores)):
            if scores[index] == elem:
                Arr.append(index)
                break
    for index in range(0,len(weight1)):
        if index != Arr[0] and index != Arr[1]:
            mix = index % 2
            l1 = weight1[index].tolist()
            l2 = weight1[Arr[mix]].tolist()
            weight1[index] = np.asarray(swap(l1,l2))
            
            l3 = bias1[index].tolist()
            l4 = bias1[Arr[mix]].tolist()
            bias1[index] = np.asarray(swap(l3,l4))
            
            l5 = weight2[index].tolist()
            l6 = weight2[Arr[mix]].tolist()
            weight2[index] = np.asarray(swap(l5,l6))
            
            l7 = bias2[index].tolist()
            l8 = bias2[Arr[mix]].tolist()
            bias2[index] = np.asarray(swap(l7,l8))
        elif index == Arr[len(Arr)-1]:
            l = weight1[Arr[0]].tolist()
            L = weight1[Arr[1]].tolist()
            weight1[index] = np.asarray(swap(l,L))
            
            l3 = bias1[Arr[0]].tolist()
            l4 = bias1[Arr[1]].tolist()
            bias1[index] = np.asarray(swap(l3,l4))
            
            l5 = weight2[Arr[0]].tolist()
            l6 = weight2[Arr[1]].tolist()
            weight2[index] = np.asarray(swap(l5,l6))
            
            l7 = bias2[Arr[0]].tolist()
            l8 = bias2[Arr[1]].tolist()
            bias2[index] = np.asarray(swap(l7,l8))
        else:
            continue
    return weight1,bias1,weight2,bias2


# In[41]:


'''
Model just with backpropagation
'''
def BP(X,Y):
    n = create_model_null()
    n.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    Count = 0
    # while Count < 10:
    # n.fit(X, Y, validation_split=0.2, epochs=100, batch_size=16, verbose=1, shuffle = True)
    history = n.fit(X, Y, validation_split=0.2, epochs=2000, batch_size=20, verbose=1, shuffle = True)
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[42]:


'''
def create_Initial():
    weight1 = np.random.rand(34,68)
    bias1 = np.random.rand(1,68)
    weight2 = np.random.rand(68,1)
    bias2 = np.random.rand(1,1)
    return weight1,bias1,weight2,bias2
'''


# In[43]:


def sigmoid(inpt):
    return 1.0 / (1.0 + np.exp(-1 * inpt))


# In[44]:


if __name__ == "__main__":
    '''
    Read data from the file and check the format about the data
    '''
    df = pd.read_csv('./ionosphere.csv')
    print(df.head())
    '''
    Modification on the data and separate them as Attribute and Target
    '''
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    Target = []
    for elem in Y:
        if elem == 'b':
            Target.append([0])
        else:
            Target.append([1])
    Target = np.asarray(Target)
    Target.transpose()
    # print(X.head())
    # print(Target)
    arr = []
    weight1 = []
    bias1 = []
    weight2 = []
    bias2 = []
    scores = []
    for count in range(10):
        w1,bia1,w2,bia2 = create_Initial()
        weight1.append(w1)
        bias1.append(bia1)
        weight2.append(w2)
        bias2.append(bia2)
    # print(scores)
    for num_generation in range(200):
        print(num_generation)
        arr = []
        cur_accur = 0
        cur_scores = []
        for index in range(len(weight1)):
            m = create_model(weight1[index],bias1[index],weight2[index],bias2[index])
            m.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
            accr = m.test_on_batch(X, Target, sample_weight=None, reset_metrics=True)[1]
            cur_accur += accr
            cur_scores.append(accr)
        
        scores.append(cur_accur/len(weight1))
        weight1,bias1,weight2,bias2 = generate_new_weight(weight1,bias1,weight2,bias2,cur_scores)
    # print(scores)
    plt.plot(scores)
    plt.show()
    
    BP(X,Target)
    


# In[ ]:





# In[ ]:




