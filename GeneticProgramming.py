#!/usr/bin/env python
# coding: utf-8

# In[305]:


import pandas as pd
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import random
import numpy as np


# In[306]:


df = pd.read_csv('./ionosphere.csv')
X = df.iloc[:,2:-1]
Y = df.iloc[:,-1]
Target = []
for elem in Y:
    if elem == 'b':
        Target.append([0])
    else:
        Target.append([1])
X


# In[307]:


function_set = {'+','-','*', '/'}


# In[308]:


L = ["x"+str(i) for i in range(1,33,1)]
terminal_set = set(L)
# Here the terminal_set havent add the R between [0,10], but generate 0.5 for class attribute and 0.5 for random int


# In[309]:


# Generate initial population
import random
class Node:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

# In-order traversal of tree
def inorder(cur):
    if not cur:
        return []
    else:
        return inorder(cur.left) + [cur.val] + inorder(cur.right)

# Create a random tree structure for initial population
def create(k):
    if k == 0:
        return None
    decide = random.randint(0,1)
    cur = None
    if decide % 2 == 0:   #If even number, generate a terminal
        terminal_prob = random.randint(0,1)
        if terminal_prob % 2 == 0:
            cur = Node(random.random())
        else:
            l = list(terminal_set)
            index = random.randint(0,len(l)-1)
            cur = Node(l[index])
    else:   #If odd number, generate a function
        l = list(function_set)
        index = random.randint(0,len(l)-1)
        cur = Node(l[index])
        cur.left = create(k-1)
        cur.right = create(k-1)
    return cur

# Valid the tree structure
def isValid(cur):
    for index in range(len(cur)):
        if index == 0 and cur[0] in function_set:
            return False
        elif index == len(cur)-1 and cur[index] in function_set:
            return False
        elif cur[index] in function_set and (cur[index+1] in function_set or cur[index-1] in function_set):
            return False
    return True

# Generate n initial population
def generate(n):
    root = []
    res = []
    while len(root) < n:
        temp = create(random.randint(1,34))
        t = inorder(temp)
        if isValid(t):
            res.append(t)
            root.append(temp)
    return res,root


# In[310]:


from math import *
# Evaluate the arithmetic expression
def evaluate(cur):
    temp = []
    for elem in cur:
        temp.append(str(elem))
    temp = "".join(temp)
    #print(temp)
    return eval(temp)


# In[311]:


# Store all those in-order travsel representation, root and calculate the corresponding fitness
def updatePopulation(population,Root):
    Root_dic = dict()
    Population_dic = dict()
    for loc,people in enumerate(population):
        count = 0
        Total = 0
        for index, row in df.iterrows():
            Total += 1
            cur = []
            for i in range(len(people)):
                if people[i] in terminal_set:
                    col_index= int(people[i][1:])
                    val = row[col_index]
                    if val == 0:
                        val = 0.00001
                    cur.append(val)
                else:
                    cur.append(people[i])
            predict = evaluate(cur)
            if predict < 0 and Target[index][0] == 0:
                count+= 1
            elif predict >= 0 and Target[index][0] == 1:
                count+= 1
            
        Root_dic[tuple(people)] = Root[loc]
        Population_dic[tuple(people)] = count / float(Total)
    
    return Root_dic, Population_dic
    


# In[312]:


# Perform the mutation operator, select node at random height, change to a random value based on the set its in
def mutation(rootPopulation:dict(),stringPopulation:dict()):
    literal = []
    root = []
    fitness = []
    for k, v in rootPopulation.items():
        literal.append(k)
        root.append(v)
        fitness.append(stringPopulation[k])
    index = random.randint(0,len(fitness)-1)
    
    to_change = root[index]
    level = random.randint(0,10)
    #print(len(literal),len(root),len(fitness))
    #print(root[index])
    def modification(cur,depth):
        if cur.left == None and cur.right == None:
            terminal_prob = random.randint(0,1)
            if terminal_prob % 2 == 0:
                cur.val = random.random()
            else:
                l = list(terminal_set)
                index = random.randint(0,len(l)-1)
                cur.val = l[index]
        elif depth > 0:
            direction = random.randint(0,1)
            if direction % 2 == 0 and cur.left:
                modification(cur.left, depth-1)
            elif direction % 2 == 1 and cur.right:
                modification(cur.right, depth-1)
            elif cur.left and not cur.right:
                modification(cur.left,depth-1)
            elif cur.right and not cur.left:
                modification(cur.right,depth-1)
        elif depth == 0:
            if cur.val in function_set:
                l = list(function_set)
                index = random.randint(0,len(l)-1)
                while l[index] == cur.val:
                    index = random.randint(0,len(l)-1)
                cur.val = l[index]
            else:
                terminal_prob = random.randint(0,1)
                if terminal_prob % 2 == 0:
                    cur.val = random.random()
                else:
                    l = list(terminal_set)
                    index = random.randint(0,len(l)-1)
                    cur.val = l[index]
    
    #print(inorder(to_change))
    modification(to_change,level)
    #print(inorder(to_change))
    return inorder(to_change), to_change


# In[313]:


# Perform the crossover operator, seclct 2 random parents, and make sure the node being swapped has same type of value
def crossover(rootPopulation:dict(),stringPopulation:dict()):
    literal = []
    root = []
    fitness = []
    for k, v in rootPopulation.items():
        literal.append(k)
        root.append(v)
        fitness.append(stringPopulation[k])
    # print(root)
    # print(len(fitness))
    index1 = random.randint(0,len(fitness)-1)
    index2 = random.randint(0,len(fitness)-1)
    while index2 == index1:
        index2 = random.randint(0,len(fitness)-1)
    
    # test
    # index1, index2 = 0, 3
    parent1 = root[index1]
    parent2 = root[index2]
    
    def modification(parent1, parent2):
        if parent1.val in function_set and parent2.val in function_set:
            chance = random.randint(0,1)
            if chance == 0:
                t = parent1.left
                parent1.left = parent2.left
                parent2.left = t
            else:
                t = parent1.right
                parent1.right = parent2.right
                parent2.right = t
            return
        elif parent1.val not in function_set and parent2.val not in function_set:
            chance = random.randint(0,1)
            if chance == 0:
                t = parent1.left
                parent1.left = parent2.left
                parent2.left = t
            else:
                t = parent1.right
                parent1.right = parent2.right
                parent2.right = t
            return
        elif parent1.val in function_set:
            if parent1.left:
                modification(parent1.left,parent2)
            else:
                modification(parent1.right,parent2)
        elif parent2.val in function_set:
            if parent2.left:
                modification(parent1,parent2.left)
            else:
                modification(parent1,parent2.right)
        
    
    modification(parent1, parent2)
    st1 = inorder(parent1)
    st2 = inorder(parent2)
    r1 = parent1
    r2 = parent2
    return st1,st2,r1,r2


# In[314]:


# The overall breeding process for generating the population for next generation with assigning reproduction, crossover
# and mutation with different probability
def breeding(rootPopulation, stringPopulation):
    p_next = 0
    stringRep = []
    rootRep = []
    while p_next < 50:
        operator = random.randint(1,10)
        if operator <= 2 :
            Str, root = generate(1)
            for index in range(len(root)):
                stringRep.append(Str[index])
                rootRep.append(root[index])
            p_next += 1
        if 2 < operator <= 6 :
            s1,s2,r1,r2 = crossover(rootPopulation,stringPopulation)
            stringRep.append(s1)
            rootRep.append(r1)
            stringRep.append(s2)
            rootRep.append(r2)
            p_next += 2
        if 7 < operator  <= 10:
            Str, root = mutation(rootPopulation,stringPopulation)
            stringRep.append(Str)
            rootRep.append(root)
            p_next += 1
    return stringRep, rootRep
    


# In[301]:


# Initial 50 population
population, Root = generate(50)
# Update the population, like get the fitness and root node for each population
rootPopulation, stringPopulation = updatePopulation(population,Root)
# Best solution
p_best_score = 0
p_best = ""
generation = 0
history = []
# print(rootPopulation, stringPopulation)
while generation < 2000 and p_best_score < 0.85:
    generation += 1
    stringRepresentation, root = breeding(rootPopulation, stringPopulation)
    #print(stringRepresentation, root, "from line 13")
    r_p, s_p = updatePopulation(stringRepresentation, root)
    for k, v in s_p.items():
        if v > p_best_score:
            p_best = k
            p_best_score = v
    rootPopulation, stringPopulation = r_p, s_p
    print(p_best_score, generation)
    history.append(p_best_score)
        


# In[302]:


plt.plot(history)


# In[303]:


print(p_best)


# In[304]:


plt.plot(history[:500])


# In[ ]:




