# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:53:32 2022

@author: DAVID MORA MEZA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Scenario(s):
    if s<=5:
        n=10 #Number of clients
        maxD=100
    else:
        n=100
        maxD=250

    

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
    
    return n, maxD, ProfitC, D, coordXY


#### Profit ####
def BigestProfit(D,ProfitC,maxD,n):
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


### Plus proche Voisin ####

def PPV(D,ProfitC,maxD,n):
    sumaD=0
    sumaP=0

    DT=D.copy() #Distance temporel
    for j in range(0,n+1): DT[j,j]=10000

    i=0
    ClientsV=[]
    while True:
        pv=np.argmin(DT[i,1:])+1
        if sumaD + D[i,pv] + D[pv,0] > maxD or len(ClientsV)==n: break
        sumaD += D[i,pv]
        sumaP += ProfitC[pv-1]
        for j in range(0,n+1): 
            DT[i,j]=10000
            DT[j,i]=10000
        ClientsV.append(pv)
        i=pv

    sumaD += D[i,0]
    
    return ClientsV ,sumaD, sumaP


### Profit/D ####

def RelProDist(D,ProfitC,maxD,n):
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
        
        if sumaD + D[i,j] + D[j,0] > maxD or len(ClientsV)==n: break
    
        sumaD += D[i,j]
        sumaP += ProfitC[j-1]
        ProfitCT[j-1]=0
        ClientsV.append(j)
        i=j

    sumaD += D[i,0]
    
    return ClientsV ,sumaD, sumaP



def PlotSolution(coordXY,ProfitC,ClientsV,Title=''):
    fig, ax= plt.subplots(figsize=(7.5,5.5))

    ax.plot(coordXY[0,0],coordXY[0,1],'.', markersize=10)
    ax.plot(coordXY[1:,0],coordXY[1:,1],'.', markersize=5)

    for i, P in enumerate(ProfitC):
        ax.text(coordXY[i+1,0],coordXY[i+1,1],P)

    Tourne=np.zeros((len(ClientsV)+2,2))
    Tourne[0] = coordXY[0]
    Tourne[-1] = coordXY[0]

    j=1
    for i in ClientsV:
        Tourne[j]=coordXY[i]
        j+=1
    
    ax.plot(Tourne[:,0],Tourne[:,1])
    
    fig.suptitle(Title, y=0.95)



def distance(ClientsV,D):
    sumaD=D[0,ClientsV[0]]+D[ClientsV[-1],0]
    for i in range(1,len(ClientsV)):
        sumaD+=D[ClientsV[i-1],ClientsV[i]]
    return sumaD

def profit(ClientsV,ProfitC):
    return sum(ProfitC[list(np.array(ClientsV)-1)])  


def M1(ClientsV,D):
    ClientsVD=[0]+ClientsV+[0]
    while True:
        deltaMin=0
        for u in range(1,len(ClientsVD)-1):
            for v in range(u+1,len(ClientsVD)-1):
                x=u-1
                y=v+1
                deltaD = D[ClientsVD[x],ClientsVD[v]] + D[ClientsVD[u],ClientsVD[y]] - D[ClientsVD[x],ClientsVD[u]] - D[ClientsVD[v],ClientsVD[y]]
                
                if deltaD <= deltaMin:
                    deltaMin=deltaD
                    bestU=u
                    bestV=v
        
        if deltaMin < -0.000001:
            CR=ClientsVD[bestU:bestV+1]
            CR.reverse()
            ClientsVD=ClientsVD[:bestU]+CR+ClientsVD[bestV+1:]
        else:
            break
    ClientsVD.remove(0)
    ClientsVD.remove(0)
    return ClientsVD


def M2(ClientsV,D,ProfitC):
    while True:
        ClientsNV=list(range(1,n+1))
        for v in ClientsV:
            ClientsNV.remove(v)
        PosibleClients=[]
        PosibleProfits=[]
        for c in ClientsNV:
            ClientsVN=ClientsV.copy()
            for j in range(0,len(ClientsV)+1):
                ClientsVN=ClientsV[0:j]+[c]+ClientsV[j:]
                if distance(ClientsVN,D) <= maxD:
                    PosibleClients.append(ClientsVN)
                    PosibleProfits.append(profit(ClientsVN,ProfitC))
        if PosibleProfits!=[]:
            ClientsV=PosibleClients[np.argmax(PosibleProfits)]
        else:
            break
    return ClientsV



def M3(ClientsV,D,ProfitC):
    
    while True:
        ClientsV1=np.array(ClientsV)
        ClientsNV=list(range(1,n+1))
        for v in ClientsV:
            ClientsNV.remove(v)
        
        PosibleClients=[]
        PosibleProfits=[]
        for nv in ClientsNV:
            better=ProfitC[list(ClientsV1-1)]<ProfitC[nv-1]
            for c in ClientsV1[better]:
                ClientsV2=ClientsV.copy()
                ClientsV2.remove(c)
                for j in range(0,len(ClientsV2)):
                    ClientsVN=ClientsV2[0:j]+[nv]+ClientsV2[j:]
                    if distance(ClientsVN,D) <= maxD:
                        PosibleClients.append(ClientsVN)
                        PosibleProfits.append(profit(ClientsVN,ProfitC))
        if PosibleProfits!=[]:
            ClientsV=PosibleClients[np.argmax(PosibleProfits)]
        else:
            break
        
    return ClientsV

#%%
Opt=[202,105,155,145,182] #Optimos trouv??s avec GAMS (Gusek)

SolsD=[]
SolsP=[]
SolsClients=[]

'''Scenario wanted, the first 5 are with 10 clients and the second 5 are with 100 clients'''
for s in range(1,6):
    n, maxD, ProfitC, D, coordXY = Scenario(s)
    print(f'Starting scenario {s}')
    ClientsV ,sumaD, sumaP=BigestProfit(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    print('Finished BigestProfit')
    
    ClientsV ,sumaD, sumaP=PPV(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    print('Finished PPV')
    
    ClientsV ,sumaD, sumaP=RelProDist(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    print('Finished RelProDist')

H=np.array([['Biggest Profit']*5,['Plus Proche Voisin']*5,['Profit/Distance']*5])
H=list(H.flatten())
df = pd.DataFrame({'Scenario':sum([[i]*15 for i in list(range(1,6))],[]),
                   'Heuristic':H*5, 
                   'Modification':['Non','M1','M1, M2','M1, M2, M3','MULT']*15, 
                   'Distance':SolsD, 'Profit':SolsP,
                   'Profit Gusek':sum([[i]*15 for i in Opt],[]),
                   'Clients Visited':SolsClients})

df['Error']=(df['Profit Gusek']-df['Profit'])/df['Profit Gusek']
df['Nombre Clients Visited']=df['Clients Visited'].apply(lambda x: len(x))

#%%
SolsD=[]
SolsP=[]
SolsClients=[]

'''Scenario wanted, the first 5 are with 10 clients and the second 5 are with 100 clients'''
for s in range(6,11):
    n, maxD, ProfitC, D, coordXY = Scenario(s)
    print(f'Starting scenario {s}')
    ClientsV ,sumaD, sumaP=BigestProfit(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    
    print('Finished BigestProfit')
    
    ClientsV ,sumaD, sumaP=PPV(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    
    print('Finished PPV')
    
    ClientsV ,sumaD, sumaP=RelProDist(D,ProfitC,maxD,n)
    SolsD.append(sumaD)
    SolsP.append(sumaP)
    SolsClients.append(ClientsV)
    ClientsV = M1(ClientsV,D)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M2(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    ClientsV = M3(ClientsV,D,ProfitC)
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    while True:
        MinProf=profit(ClientsV,ProfitC)
        ClientsV = M1(ClientsV,D)
        ClientsV = M2(ClientsV,D,ProfitC)
        ClientsV = M3(ClientsV,D,ProfitC)
        if profit(ClientsV,ProfitC)==MinProf: break
    SolsD.append(distance(ClientsV,D))
    SolsP.append(profit(ClientsV,ProfitC))
    SolsClients.append(ClientsV)
    
    print('Finished RelProDist')

H=np.array([['Biggest Profit']*5,['Plus Proche Voisin']*5,['Profit/Distance']*5])
H=list(H.flatten())
dfB = pd.DataFrame({'Scenario':sum([[i]*15 for i in list(range(6,11))],[]),
                   'Heuristic':H*5, 
                   'Modification':['Non','M1','M1, M2','M1, M2, M3','MULT']*15, 
                   'Distance':SolsD, 'Profit':SolsP,
                   'Clients Visited':SolsClients})
dfB['Nombre Clients Visited']=dfB['Clients Visited'].apply(lambda x: len(x))

#%%
sns.set_style("darkgrid")
sns.set_palette('colorblind')
sns.set_context('notebook')

f=sns.catplot(x='Modification', y="Error", hue='Heuristic',data=df, kind="point",
              capsize=0.2,join=True,ci='sd',col="Heuristic") 

f.fig.suptitle("Moyenne et ??cart-type de l'erreur de l'heuristique en 5 petits cas",y=1.01, size=15)

f.fig.set_figwidth(10)
f.fig.set_figheight(7)
f.set_xticklabels(rotation=30)
#f.set(xlabel="New X Label", ylabel="New Y Label")

f.savefig('Error 5 petit cases and modifications.png', quality=500)

plt.show()

#%%
sns.set_style("darkgrid")
sns.set_palette('colorblind')
sns.set_context('notebook')

f=sns.catplot(x='Modification', y="Profit", hue='Scenario',data=df, kind="point",
              capsize=0.2,join=True,ci='sd',col="Heuristic") 

f.fig.suptitle("Profit de l'heuristique en 5 petits cas",y=1.01, size=15)

f.fig.set_figwidth(10)
f.fig.set_figheight(7)
f.set_xticklabels(rotation=30)
#f.set(xlabel="New X Label", ylabel="New Y Label")

f.savefig('Profit 5 petit cases and modifications.png', quality=500)

plt.show()

#%%
sns.set_style("darkgrid")
sns.set_palette('colorblind')
sns.set_context('notebook')

f=sns.catplot(x='Modification', y="Profit", hue='Heuristic',data=dfB, kind="point",
              capsize=0.2,join=True,ci='sd',col="Heuristic") 

f.fig.suptitle("Moyenne et ??cart-type du profit de l'heuristique en 5 grands cas",y=1.01, size=15)

f.fig.set_figwidth(10)
f.fig.set_figheight(7)
f.set_xticklabels(rotation=30)
#f.set(xlabel="New X Label", ylabel="New Y Label")

f.savefig('Mean Profit 5 grand cases and modifications.png', quality=500)

plt.show()

#%%
sns.set_style("darkgrid")
sns.set_palette('colorblind')
sns.set_context('notebook')

f=sns.catplot(x='Modification', y="Profit", hue='Scenario',data=dfB, kind="point",
              capsize=0.2,join=True,ci='sd',col="Heuristic") 

f.fig.suptitle("Profit de l'heuristique dans 5 grands cas",y=1.01, size=15)

f.fig.set_figwidth(10)
f.fig.set_figheight(7)
f.set_xticklabels(rotation=30)
#f.set(xlabel="New X Label", ylabel="New Y Label")

f.savefig('Profit 5 grand cases and modifications.png', quality=500)

plt.show()

#%%
def PlotMultipleSolution(ax,coordXY,ProfitC,ClientsV,Title=''):
    
    ax.plot(coordXY[0,0],coordXY[0,1],'.', markersize=10)
    ax.plot(coordXY[1:,0],coordXY[1:,1],'.', markersize=5)
    
    for i, P in enumerate(ProfitC):
        ax.text(coordXY[i+1,0],coordXY[i+1,1],P)
    
    Tourne=np.zeros((len(ClientsV)+2,2))
    Tourne[0] = coordXY[0]
    Tourne[-1] = coordXY[0]

    j=1
    for i in ClientsV:
        Tourne[j]=coordXY[i]
        j+=1
    
    ax.plot(Tourne[:,0],Tourne[:,1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(Title,y=0.98)
    

#%%
s=4
n, maxD, ProfitC, D, coordXY = Scenario(s)
ClientsV ,sumaD, sumaP=PPV(D,ProfitC,maxD,n)

fig = plt.figure(figsize=(11,9))
ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax3=fig.add_subplot(3,2,3)
ax4=fig.add_subplot(3,2,4)
ax5=fig.add_subplot(3,2,(5,6))

fig.suptitle(f'Scenario {s} d??velopp?? avec Plus Proche Voisin avec les diff??rentes modifications',y=0.94)

PlotMultipleSolution(ax1,coordXY,ProfitC,ClientsV,f'Modification: Non, Profit: {sumaP}, Distance: {round(sumaD,2)}')

ClientsV=M1(ClientsV,D)
SumaDN=distance(ClientsV,D)
PlotMultipleSolution(ax2,coordXY,ProfitC,ClientsV,f'Modification: M1, Profit: {sumaP}, Distance: {round(SumaDN,2)}')
ClientsV=M2(ClientsV,D,ProfitC)
sumaP=profit(ClientsV, ProfitC)
PlotMultipleSolution(ax3,coordXY,ProfitC,ClientsV,f'Modification: M2, Profit: {sumaP}, Distance: {round(distance(ClientsV,D),2)}')
ClientsV=M3(ClientsV,D,ProfitC)
sumaP=profit(ClientsV, ProfitC)
PlotMultipleSolution(ax4,coordXY,ProfitC,ClientsV,f'Modification: M3, Profit: {sumaP}, Distance: {round(distance(ClientsV,D),2)}')
while True:
    MinProf=profit(ClientsV,ProfitC)
    ClientsV = M1(ClientsV,D)
    ClientsV = M2(ClientsV,D,ProfitC)
    ClientsV = M3(ClientsV,D,ProfitC)
    if profit(ClientsV,ProfitC)==MinProf: break

PlotMultipleSolution(ax5,coordXY,ProfitC,ClientsV,f'Modification: Multiple fois, Profit: {profit(ClientsV, ProfitC)}, Distance: {round(distance(ClientsV,D),2)}')
fig.savefig(f'Scenario {s} d??velopp?? avec Plus Proche Voisin avec des modifications.png',bbox_inches='tight', pad_inches=0.15, dpi=500)

#%%
s=8
n, maxD, ProfitC, D, coordXY = Scenario(s)
ClientsV ,sumaD, sumaP=BigestProfit(D,ProfitC,maxD,n)

fig = plt.figure(figsize=(11,9))
ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax3=fig.add_subplot(3,2,3)
ax4=fig.add_subplot(3,2,4)
ax5=fig.add_subplot(3,2,(5,6))

fig.suptitle(f'Scenario {s} d??velopp?? avec Biggest Profit avec les diff??rentes modifications',y=0.94)

PlotMultipleSolution(ax1,coordXY,ProfitC,ClientsV,f'Modification: Non, Profit: {sumaP}, Distance: {round(sumaD,2)}')

ClientsV=M1(ClientsV,D)
SumaDN=distance(ClientsV,D)
PlotMultipleSolution(ax2,coordXY,ProfitC,ClientsV,f'Modification: M1, Profit: {sumaP}, Distance: {round(SumaDN,2)}')
ClientsV=M2(ClientsV,D,ProfitC)
sumaP=profit(ClientsV, ProfitC)
PlotMultipleSolution(ax3,coordXY,ProfitC,ClientsV,f'Modification: M2, Profit: {sumaP}, Distance: {round(distance(ClientsV,D),2)}')
ClientsV=M3(ClientsV,D,ProfitC)
sumaP=profit(ClientsV, ProfitC)
PlotMultipleSolution(ax4,coordXY,ProfitC,ClientsV,f'Modification: M3, Profit: {sumaP}, Distance: {round(distance(ClientsV,D),2)}')
while True:
    MinProf=profit(ClientsV,ProfitC)
    ClientsV = M1(ClientsV,D)
    ClientsV = M2(ClientsV,D,ProfitC)
    ClientsV = M3(ClientsV,D,ProfitC)
    if profit(ClientsV,ProfitC)==MinProf: break

PlotMultipleSolution(ax5,coordXY,ProfitC,ClientsV,f'Modification: Multiple fois, Profit: {profit(ClientsV, ProfitC)}, Distance: {round(distance(ClientsV,D),2)}')
fig.savefig(f'Scenario {s} d??velopp?? avec Biggest Profit avec des modifications.png',bbox_inches='tight', pad_inches=0.15, dpi=500)

#%%

''' GUSEK Solutions '''
solsG=[[8,1,2,10,6,5,9,4],[3,10,9],[6,4,7,1,3,5,9],[4,9,6,7,3],[3,2,5,7,1,10,8]]
ProfG=[]
DistG=[]
for i in range(0,len(solsG)):
    s=i+1
    n, maxD, ProfitC, D, coordXY = Scenario(s)
    
    ProfG.append(profit(solsG[i], ProfitC))
    DistG.append(distance(solsG[i],D))
    PlotSolution(coordXY,ProfitC,solsG[i],f'Gusek Solution, Scenario {s} Profit Total: {ProfG[i]}, Distance: {round(DistG[i],3)}')