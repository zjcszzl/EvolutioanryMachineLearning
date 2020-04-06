#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random
import numpy as np
import keras
import tensorflow


# In[2]:


'''
Create a model without assigning the Model Weight, and the Model has only 1 hidden layer with twice as many nodes as the input dimensionality
'''
def create_model_null():
    model = Sequential()
    model.add(Dense(68,input_dim = 34, activation = 'sigmoid'))
    model.add(Dense(1,activation = 'sigmoid'))
    return model


# In[3]:


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


# In[4]:


def create_Initial():
    weight1 = np.random.rand(34,68)
    bias1 = np.random.rand(1,68)
    weight2 = np.random.rand(68,1)
    bias2 = np.random.rand(1,1)
    return weight1,bias1,weight2,bias2


# In[5]:


'''
    Evolution Strategy:
        parent randomly chosen from current pupolation
        Survial conditions: (u,r): only offsping may survive into next generation
                            (u+r): both the previous and offspring are considered
        u: often 1 - 3
        r: many more offspring are generated than the population size, r = 100 when u =3
        
        mutaton: no mutation probability, but use Gaussian distributions
        small amount of recombination possible, not necessary
        
        each individual may associated with a separater parameter value(mutation rate), which is evolved along with
        the individual
'''


# In[6]:


def sigmoid(inpt):
    return 1.0 / (1.0 + np.exp(-1 * inpt))


# In[7]:


'''
    Return the accuracy of the NN model with assigned weight
'''
def fitness_function(weight1,bias1,weight2,bias2):
    m = create_model(weight1,bias1,weight2,bias2)
    m.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    accr = m.test_on_batch(X, Target, sample_weight=None, reset_metrics=True)[1]
    return accr


# In[8]:


'''
    Return the new created weight and bias for children after mutation with Gaussian Distribution
'''
def normal_distribution_affect(weight1,bias1,weight2,bias2,sig):
    for row in weight1:
        for col in row:
            col += np.random.normal(0, sig, 1)
    for row in bias1:
        for col in row:
            col += np.random.normal(0, sig, 1)
    for row in weight2:
        for col in row:
            col += np.random.normal(0, sig, 1)
    for row in bias2:
        for col in row:
            col += np.random.normal(0, sig, 1)
    return weight1,bias1,weight2,bias2


# In[9]:


'''
    Add the newly recombiened children from different parent to the arr. And the value of children is 
    by average the value of its parents
'''
def create_recombination_child(weight1,bias1,weight2,bias2,mutation_sig,success_count):
    size = len(weight1)
    p1 = random.randint(0,size-1)
    p2 = random.randint(0,size-1)
    while p1 == p2:
        p2 = random.randint(0,size-1)
    w1,b1,w2,b2 = create_Initial()
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            w1[i][j] = (weight1[p1][i][j] + weight1[p2][i][j]) / 2
    for i in range(len(b1)):
        for j in range(len(b1[i])):
            b1[i][j] = (bias1[p1][i][j] + bias1[p2][i][j]) / 2
    for i in range(len(w2)):
        for j in range(len(w2[i])):
            w2[i][j] = (weight2[p1][i][j] + weight2[p2][i][j]) / 2
    for i in range(len(b2)):
        for j in range(len(b2[i])):
            b2[i][j] = (bias2[p1][i][j] + bias2[p2][i][j]) / 2
    mutation_sig.append( (mutation_sig[p1] + mutation_sig[p2]) / 2)
    success_count.append(0)
    weight1.append(w1)
    bias1.append(b1)
    weight2.append(w2)
    bias2.append(b2)
    return weight1,bias1,weight2,bias2,mutation_sig,success_count


# In[13]:


'''
    Using the function created to get the mutated weight and bias to create the children from the mutation
    process, 9 children is returned. Also, based on the mutated result, record the performance of certain
    mutation rate
'''
def create_mutation_child(weight1,bias1,weight2,bias2,mutation_sig,suceess_count):
    w1 = weight1.copy()
    b1 = bias1.copy()
    w2 = weight2.copy()
    b2 = bias2.copy()
    s1 = success_count.copy()
    for i in range(9):
        index = random.randint(0,len(weight1)-1)
        new_weight1,new_bias1,new_weight2, new_bias2 = normal_distribution_affect(weight1[index],bias1[index],weight2[index],bias2[index],mutation_sig[index])
        if(fitness_function(new_weight1,new_bias1,new_weight2,new_bias2) > fitness_function(weight1[index],bias1[index],weight2[index],bias2[index])):
            suceess_count[index] += 1
        w1.append(new_weight1)
        b1.append(new_bias1)
        w2.append(new_weight2)
        b2.append(new_bias2)
        mutation_sig.append(mutation_sig[index])
        success_count.append(success_count[index])
    print(len(w1),len(b1),len(w2),len(b2),len(mutation_sig))
    return w1,b1,w2,b2,mutation_sig,success_count
    


# In[14]:


'''
    Self-defined servival function for the offspring, first of all, no parent is being passed into this function
    and by sorting the Accuracy of all offsprings, the best 3 offspring is returned.
'''
def servivor_selection(weight1,bias1,weight2,bias2,mutation_sig,success_count):
    print(len(weight1),len(bias1),len(weight2),len(bias2),len(mutation_sig))
    score_holder = []
    for index in range(len(weight1)):
        score_holder.append(fitness_function(weight1[index],bias1[index],weight2[index],bias2[index]))
    score = score_holder.copy()
    score.sort(reverse = True)
    nw1 = []
    nb1 = []
    nw2 = []
    nb2 = []
    mut_sig = []
    sc = []
    for index in range(len(weight1)):
        if score_holder[index] == score[0]:
            nw1.append(weight1[index])
            nb1.append(bias1[index])
            nw2.append(weight2[index])
            nb2.append(bias2[index])
            mut_sig.append(mutation_sig[index])
            sc.append(success_count[index])
        elif score_holder[index] == score[1]:
            nw1.append(weight1[index])
            nb1.append(bias1[index])
            nw2.append(weight2[index])
            nb2.append(bias2[index])
            mut_sig.append(mutation_sig[index])
            sc.append(success_count[index])
        elif score_holder[index] == score[2]:
            nw1.append(weight1[index])
            nb1.append(bias1[index])
            nw2.append(weight2[index])
            nb2.append(bias2[index])
            mut_sig.append(mutation_sig[index])
            sc.append(success_count[index])
    return nw1[:3],nb1[:3],nw2[:3],nb2[:3],mut_sig[:3],sc[:3],score[0]
    


# In[15]:


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
    
    mutation_sig = []
    success_count = []
    for count in range(3):
        w1,bia1,w2,bia2 = create_Initial()
        weight1.append(w1)
        bias1.append(bia1)
        weight2.append(w2)
        bias2.append(bia2)
        mutation_sig.append(random.randint(1,10))
        success_count.append(0)
    print(len(mutation_sig))
    
    score_h = 0
    score = []
    count = 0
    num = 0
    while score_h < 0.9 and num < 500:
        num += 1
        count += 1
        # for i in range(2):
        weight1,bias1,weight2,bias2,mutation_sig,success_count = create_recombination_child(weight1,bias1,weight2,bias2,mutation_sig,success_count)
        weight1,bias1,weight2,bias2,mutation_sig,success_count = create_mutation_child(weight1,bias1,weight2,bias2,mutation_sig,success_count)
        weight1,bias1,weight2,bias2,mutation_sig,success_count,score_h = servivor_selection(weight1[3:],bias1[3:],weight2[3:],bias2[3:],mutation_sig[3:],success_count[3:])
        if count == 20:
            for index in range(len(success_count)):
                if success_count[index] >= 4:
                    mutation_sig[index] = mutation_sig[index] / 0.8
                    success_count[index] = 0
                else:
                    mutation_sig[index] = mutation_sig[index] * 0.8
                    success_count[index] = 0
            count = 0
        print(mutation_sig)
        print(success_count)
        print(score_h)
        print(count,num)
        score.append(score_h)
        
    
    
    


# In[27]:


plt.plot(score)
plt.ylim(0.55,0.85)
plt.show()


# In[ ]:




