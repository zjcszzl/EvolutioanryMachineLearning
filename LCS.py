import numpy as np
import pandas as pd
import xcs
from xcs import XCSAlgorithm
import random
import matplotlib.pyplot as plt


# Based on the ramdon assignment of 0 ,1 and #, generate the rule
def generateRule(zeros, ones, wildcards):
    new_rule = ['0'] * zeros + ['1'] * ones + ['#'] * wildcards
    random.shuffle(new_rule)
    flag = random.randint(0, 4)
    if flag % 2 == 0:
        new_rule.append('b')
    else:
        new_rule.append('g')
    return new_rule


# Generate a random rules set of size 20
def generateRulesSet(number_of_instance):
    Rules = set()
    while True:
        if len(Rules) < 20:
            zeros = random.randint(0, 10)
            ones = random.randint(0, 10)
            wildcards = number_of_instance - zeros - ones
            if wildcards < 0:
                continue
            else:
                rule = generateRule(zeros, ones, wildcards)
                rule = tuple(rule)
                if rule not in Rules:
                    Rules.add(rule)
        else:
            break
    return Rules


# Read the dataset file and generate the list format, and separate them to train set and test set
def readFile(path):
    df = pd.read_csv(path)
    dataset = df.iloc[:, :]
    dataset = dataset.values.tolist()
    random.shuffle(dataset)
    test = dataset[0::5]
    train = dataset[1::5] + dataset[2::5] + dataset[3::5] + dataset[4::5]
    #train = dataset[:]
    #test = train
    return train, test


# Specify that negative values means 0 and positive values means 1
def checkMathAndFit(dataset, rule):
    # print(rule)
    number_of_match = 0
    number_of_fit = 0
    for row in range(len(dataset)):
        flag = 1
        #print(len(dataset[row]), len(rule))
        for col in range(len(dataset[row])-1):
            if rule[col] == '#':
                continue
            elif rule[col] == '0' and dataset[row][col] <= 0:
                continue
            elif rule[col] == '1' and dataset[row][col] > 0:
                continue
            else:
                flag = 0
                break
        if flag == 1:
            number_of_match += 1
            if dataset[row][-1] == rule[-1]:
                number_of_fit += 1
    return number_of_match, number_of_fit


# update the Rules Set, discard all the rule that have less accurafy of 0.6
def updateRules(rules_fitness, rules):
    temp = set()
    for rule, fitness in rules_fitness.items():
        if fitness[1] < 0.6:
            temp.add(rule)
    for rule in temp:
        rules_fitness.pop(rule)
        rules.remove(rule)
    return rules_fitness, rules
    # return NotImplementedError


if __name__ == "__main__":
    dataset, testset = readFile('ionosphere.csv')
    number_of_instance = len(dataset[0]) - 1
    Rules = generateRulesSet(number_of_instance)
    # print(Rules)
    Rules = list(Rules)
    rules_fitness = dict()
    for rule in Rules:
        match, fit = checkMathAndFit(dataset, rule)
        if match != 0:
            rules_fitness[rule] = [match, fit / match]
        else:
            rules_fitness[rule] = [0, 0]

    rules_fitness, rules = updateRules(rules_fitness, set(Rules))

    # Every generation, add some rules to the rules set
    def addRulesToSet(number_of_instance):
        if len(rules) >= 17:
            return
        temp = generateRulesSet(number_of_instance)
        for rule in temp:
            if rule not in rules:
                m_t, f_t = checkMathAndFit(dataset, rule)
                if m_t != 0 and f_t / m_t > 0.3:
                    rules.add(rule)
                    rules_fitness[rule] = [m_t, f_t / m_t]
        if len(rules) < 17:
            addRulesToSet(number_of_instance)

    # Every generation, choose one rule in the rules set to mutate
    def mutationStep():
        temp = list(rules)
        if len(temp) < 1:
            return
        index = random.randint(0, len(temp)-1)
        cur_rule = list(temp[index])
        loc = random.randint(0, len(cur_rule)-1)
        if cur_rule[loc] == '0' or cur_rule[loc] == '1':
            cur_rule[loc] = '#'
        elif cur_rule[loc] == '#' and loc % 2 == 0:
            cur_rule[loc] = '0'
        elif cur_rule[loc] == '#' and loc % 2 == 1:
            cur_rule[loc] = '1'
        if tuple(cur_rule) not in rules:
            m_t, f_t = checkMathAndFit(dataset, tuple(cur_rule))
            if m_t != 0:
                rules.add(tuple(cur_rule))
                rules_fitness[tuple(cur_rule)] = [m_t, f_t / m_t]
                if cur_rule[-1] == 'b':
                    return
                min_fit = f_t
                min_elem = tuple(cur_rule)
                for k, v in rules_fitness.items():
                    if v[1] * v[0] < min_fit and k[-1] != 'b':
                        min_elem = k
                        min_fit = v[1] * v[0]
                rules_fitness.pop(min_elem)
                rules.remove(min_elem)

    # Every generation, choose the best two rules in the rules set to crossover and generate the child
    def crossoverStep():
        temp = list(rules)
        if len(temp) < 2:
            return
        parent1 = random.randint(0, len(temp)-1)
        parent2 = random.randint(0, len(temp)-1)
        while parent2 == parent1:
            parent2 = random.randint(0, len(temp)-1)
        par1 = list(temp[parent1])
        par2 = list(temp[parent2])
        child = []
        index = random.randint(0, len(par1)-1)
        child = par1[:index] + par2[index:-1]
        '''
        for index in range(len(par1)-1):
            if par1[index] == '#' or par2[index] == '#':
                child.append('#')
            elif index % 2 == 0:
                child.append(par1[index])
            else:
                child.append(par2[index])
        '''
        fit_par1 = rules_fitness[tuple(list(rules)[parent1])][1]
        fit_par2 = rules_fitness[tuple(list(rules)[parent2])][1]
        if fit_par1 > fit_par2:
            child.append(par1[-1])
            # rules_fitness.pop(tuple(list(rules)[parent2]))
            # rules.remove(tuple(list(rules)[parent2]))
        else:
            child.append(par2[-1])
            # rules_fitness.pop(tuple(list(rules)[parent1]))
            # rules.remove(tuple(list(rules)[parent1]))
        if tuple(child) not in rules:
            m_t, f_t = checkMathAndFit(dataset, tuple(child))
            if m_t != 0:
                rules.add(tuple(child))
                print(f_t/m_t)
                rules_fitness[tuple(child)] = [m_t, f_t / m_t]

                if child[-1] == 'b':
                    return

                min_fit = f_t
                min_elem = tuple(child)
                for k, v in rules_fitness.items():
                    if v[1] * v[0] < min_fit and k[-1] != 'b':
                        min_elem = k
                        min_fit = v[1] * v[0]
                rules_fitness.pop(min_elem)
                rules.remove(min_elem)

    # Check the accuracy of current rules set for the test set

    def checkAccuracy():
        number_of_match = 0
        number_of_fit = 0
        for row in testset:
            temp = list(rules)
            cur = [0, 0]
            for rule in temp:
                flag = 1
                for index in range(len(rule)-1):
                    if rule[index] == '#':
                        continue
                    elif rule[index] == '0' and row[index] <= 0:
                        continue
                    elif rule[index] == '1' and row[index] > 0:
                        continue
                    else:
                        flag = 0
                        break
                if flag == 1:
                    if rule[-1] == 'b':
                        cur[0] += 1  # rules_fitness[rule][1]
                    else:
                        cur[1] += 1  # rules_fitness[rule][1]
            if cur[0] == 0 and cur[1] == 0:
                continue
            else:
                # print(cur)
                number_of_match += 1
                predict = ''
                if cur[0] > cur[1]:
                    predict = 'b'
                else:
                    predict = 'g'
                if predict == row[-1]:
                    #print(predict, row[-1])
                    number_of_fit += 1
        return (number_of_match, number_of_fit)

    # Check the subset of a string
    def isSub(x, y):
        for index in range(len(x)):
            if x[index] == y[index]:
                continue
            elif y[index] == '#':
                continue
            else:
                return False
        return True

    # Remove the redunt or over-fitting rules from the rules set
    def cleanRule():

        temp = set()
        for k, v in rules_fitness.items():
            if v[1] < 0.8 and k[-1] == 'g':
                temp.add(k)
            if v[1] * v[0] < 50 and k[-1] == 'b':
                temp.add(k)
            if v[1] < 0.85 and v[0] > 50:
                temp.add(k)
        for rule in temp:
            rules_fitness.pop(rule)
            rules.remove(rule)

        cur = list(rules)
        D = dict()
        for i in range(len(cur)-1):
            for j in range(i+1, len(cur)):
                if isSub(cur[i], cur[j]):
                    # print("aaaaaa")
                    if cur[j] in D:
                        D[cur[j]].append(cur[i])
                    else:
                        D[cur[j]] = [cur[i]]
                elif isSub(cur[j], cur[i]):
                    # print("bbbbbb")
                    if cur[i] in D:
                        D[cur[i]].append(cur[j])
                    else:
                        D[cur[i]] = [cur[j]]
                else:
                    # print("hhhhhh")
                    if cur[i] not in D:
                        D[cur[i]] = []
                    if cur[j] not in D:
                        D[cur[j]] = []
        New_rules = set()
        for k, v in rules_fitness.items():
            if k in D:
                New_rules.add(k)
        New_fitness_rules = dict()
        for k in D:
            # print(k)
            New_fitness_rules[k] = rules_fitness[k]
        return New_rules, New_fitness_rules

    # print(len(rules))
    accuracy_arr = []
    number_matched = []
    number_fitted = []
    for generation in range(1500):
        if len(rules) < 10:
            addRulesToSet(number_of_instance)
        if generation > 700:
            rules, rules_fitness = cleanRule()
        updateRules(rules_fitness, rules)
        if generation % 100 == 0:
            mutationStep()
        crossoverStep()
        # print(len(rules))
        # rules, rules_fitness = cleanRule()
        # print(len(rules))
        matched, fitted = checkAccuracy()
        if matched != 0:
            accuracy_arr.append(fitted/matched)
            number_matched.append(matched/len(testset))
            number_fitted.append(fitted/len(testset))
            score = fitted / matched
        else:
            accuracy_arr.append(0)
            number_matched.append(0)
            number_fitted.append(0)
            score = 0
        print("The accuracy of current rules set for the testset is: " +
              str(score) + " number of fitted: " + str(fitted) + " number of matched: " + str(matched))
        # print(rules_fitness)
        print(len(rules), len(rules_fitness))
        # mutationStep()
        # crossoverStep()
    plt.plot(accuracy_arr)
    plt.plot(number_matched)
    plt.plot(number_fitted)
    plt.show()
    plt.plot(accuracy_arr)
    plt.show()
    print(rules_fitness)
