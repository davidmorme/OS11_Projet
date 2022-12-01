# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:53:32 2022

@author: DAVID MORA MEZA
"""

import numpy as np
import pandas as pd


'''#### Just Fill s  ####'''
s=1 #Scenario wanted, the first 5 are with 10 clients and the second 5 are with 100 clients



if s<=5:
    n=10 #Number of clients
else:
    n=100

seeds=[111*i for i in range(1,11)]

np.random.seed(seeds[s-1])
coordXY=np.random.randint(50, size=(n+1,2))


np.random.seed(seeds[s-1]+50)
ProfitC=np.random.randint(1,50, size=n)

#Calculer la Distance
D=np.array([[np.sqrt((coordXY[i,0]-coordXY[j,0])**2+(coordXY[i,1]-coordXY[j,1])**2) for i in range(0,n+1)] for j in range(0,n+1)])

'''
#Autre manner de calculer la distance
D=np.zeros((n+1,n+1))
for i in range(0,n+1):
    for j in range(0,n+1):
        D[i,j]=np.sqrt((coordXY[i,0]-coordXY[j,0])**2+(coordXY[i,1]-coordXY[j,1])**2)
'''

pd.DataFrame(ProfitC).to_csv('ProfitsClients.csv')
pd.DataFrame(D).to_csv('Distances.csv')


#%%

order=np.argsort(-ProfitC)

maxD=100
sumaD=D[0,order[0]]
sumaP=ProfitC[order[0]]
i=1
while i<n:
    sumaD += D[order[i-1]+1,order[i]+1]
    if sumaD > maxD: break
    sumaP += ProfitC[order[i]]
    i+=1
    
ClientsV=order[0:i]+1


#%%

