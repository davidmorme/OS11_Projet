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
    for j in range(i+1,n+1):
        D[i,j]=np.sqrt((coordXY[i,0]-coordXY[j,0])**2+(coordXY[i,1]-coordXY[j,1])**2)
        D[j,i]=D[i,j]
'''

pd.DataFrame(ProfitC).to_csv('ProfitsClients.csv')
pd.DataFrame(D).to_csv('Distances.csv')


#%%
#### Profit ####
order=np.argsort(-ProfitC)

maxD=100
sumaD=0
sumaP=0
ClientsV=[]
i=0
k=0
for k in range(0,n):
    j=order[k]+1
    if sumaD + D[i,j]+D[j,0] > maxD: continue
    sumaD += D[i,j]
    sumaP += ProfitC[j-1]
    i=j
    ClientsV.append(j)

sumaD += D[i,0]

#%%

### Plus proche Voisin ####

maxD=100
sumaD=0
sumaP=0


DT=D.copy() #Distance temporel
for j in range(0,n+1): DT[j,j]=100

i=0
ClientsV=[]
while True:
    pv=np.argmin(DT[i])
    if sumaD + D[i,pv] + D[pv,0] > maxD: break
    sumaD += D[i,pv]
    sumaP += ProfitC[pv-1]
    for j in range(0,n+1): 
        DT[i,j]=100 
        DT[j,i]=100
    ClientsV.append(pv)
    i=pv

sumaD += D[i,0]
#%%

### Profit/D ####

maxD=100
sumaD=0
sumaP=0


ProfitCT=ProfitC.copy() #Profit temporel
DT=D.copy() #Distance temporel
for j in range(0,n+1): DT[j,j]=1000

i=0
ClientsV=[]
while True:
    Coef=ProfitCT/DT[i,1:]
    order=np.argsort(-Coef)
    
    for k in range(0,sum(ProfitCT>0)):
        j=order[k]+1
        if sumaD + D[i,j] + D[j,0] <= maxD: break
        
    if sumaD + D[i,j] + D[j,0] > maxD: break
    
    sumaD += D[i,j]
    sumaP += ProfitC[j-1]
    ProfitCT[j-1]=0
    ClientsV.append(j)
    i=j

sumaD += D[i,0]
