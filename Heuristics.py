# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:53:32 2022

@author: DAVID MORA MEZA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def Scenario(s):
    if s<=5:
        n=10 #Number of clients
    else:
        n=100

    maxD=10*n

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
    
    return n, maxD, ProfitC, D

#%%
#### Profit ####
def BigestProfit():
    order=np.argsort(-ProfitC)
    
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
    
    return ClientsV ,sumaD, sumaP
#%%

### Plus proche Voisin ####

def PPV():
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
    
    return ClientsV ,sumaD, sumaP
#%%

### Profit/D ####

def RelProDist():
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
    
    return ClientsV ,sumaD, sumaP

#%%
def error(sumaP):
    return (Opt[s-1]-sumaP)/Opt[s-1]


#%%
Opt=[202,105,155,145,182] #Optimos trouvés avec GAMS (Gusek)

SolsD=[]
SolsP=[]

'''Scenario wanted, the first 5 are with 10 clients and the second 5 are with 100 clients'''
for s in range(1,6):
    n, maxD, ProfitC, D = Scenario(s)

    ClientsV ,sumaD, sumaP=BigestProfit()
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    ClientsV ,sumaD, sumaP=PPV()
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    ClientsV ,sumaD, sumaP=RelProDist()
    SolsD.append(sumaD)
    SolsP.append(sumaP)

df = pd.DataFrame({'Heuristic':['Biggest Profit','Plus Proche Voisin','Profit/Distance']*5, 
                   'Distancia':SolsD, 'Profit':SolsP, 
                   'Optimo':sum([[i]*3 for i in Opt],[])})

df['Error']=(df['Optimo']-df['Profit'])/df['Optimo']

#%%
sns.set_style("darkgrid")
sns.set_palette('colorblind')
sns.set_context('notebook')

f=sns.catplot(x="Heuristic", y="Error", hue='Heuristic',data=df, kind="point",
              capsize=0.2,join=False,ci='sd') 

f.fig.suptitle("Mean and Standard Error of Heuristics in 5 cases",y=1.01, size=15)

f.fig.set_figwidth(10)
f.fig.set_figheight(7)

#f.set(xlabel="New X Label", ylabel="New Y Label")

f.savefig('Error 5 petit cases.png', quality=100)

plt.show()

