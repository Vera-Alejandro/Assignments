# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class lever:
    def __init__(self):
        # A lever has a distribution in which it gives rewards
        self.mu = np.random.uniform(-5,5,1)
        self.sd = np.random.uniform(0.1,3,1)
        
        self.Qa = 0
        self.Na = 0
        
        self.rewards = []
    
    def act(self):
        reward = np.random.normal(self.mu, self.sd, 1)
        self.rewards.append(reward)
        self.Na = self.Na + 1
        self.Qa = self.Qa + (1/self.Na)*(reward - self.Qa)
        

def plotlevers(levers):
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    
    colors = ["dodgerblue", "orange", "deeppink", "dimgrey", "lightcoral",
              "goldenrod", "limegreen", "lightseagreen", "magenta", "chocolate"]
    labels = ["Bandit1", "Bandit2", "Bandit3", "Bandit4", "Bandit5",
              "Bandit6", "Bandit7", "Bandit8", "Bandit9", "Bandit10"]
    
    plt.figure(figsize=(10,7), dpi=80)
    i = 0
    for l in levers:
        sns.distplot(l.rewards, color = colors[i], label=labels[i], **kwargs)
        i = i+1
    plt.xlim(-15, 15)
    plt.legend
    plt.show()
    plt.clf()
    

def sortingFunction(l):
    return l.Qa

def run(nlevers = 10, epsilon=0.1, iterations = 100000):
    levers = []
    for i in range(nlevers):
        l = lever()
        levers.append(l)
    
    for i in range(iterations):
        if(np.random.rand() < epsilon):
            l = np.random.choice(levers)
        else:
            templevers = levers.copy()
            np.random.shuffle(templevers) #Shuffling breaks ties randomly
            templevers.sort(reverse=True, key=sortingFunction)
            l = templevers[0]
        l.act()
    plotlevers(levers)
    for l in levers:
        print("mu=", l.mu, ", Qa=", l.Qa, ", Na=", l.Na, sep="")
    
if __name__ == '__main__':
    run()