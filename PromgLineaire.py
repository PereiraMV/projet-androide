# -*- coding: utf-8 -*-



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import *
import random as rd
from generation import *
from gurobipy import *

import time




"""
G=nx.Graph()

G.add_nodes_from(i for i in range (5))
G.add_edge(1,2,weight=1)
G.add_edge(2,3,weight=1)
G.add_edge(3,4,weight=10)
G.add_edge(4,0,weight=10)
G.add_edge(0,1,weight=10)


G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

G.add_weighted_edges_from([(0, 8,1), (0, 9,1), (0, 2,1), (1, 7,1), (2, 8,1), (2, 6,1), (2, 7,1), (3, 9,1), (3, 5,1), (3, 6,1), (4, 9,1), (5, 7,1)])
"""



  


def creervar(G,m,param=True):
   varEdges=dict()
   varNode=dict()
   listEdges=G.edges(data=True)
   listnodes=G.nodes()
   for i in listEdges:
      if (param):
         varEdges[(i[0],i[1])]=(m.addVar(vtype=GRB.BINARY,name="x"+str(i[0])+"_"+str(i[1]) ,lb=0 ),i[2]['weight'])
         m.update()
      else:
         varEdges[(i[0],i[1])]=(m.addVar(vtype=GRB.CONTINUOUS,name="x"+str(i[0])+"_"+str(i[1]) ,lb=0 ),i[2]['weight'])
         m.update()
      
   for i in listnodes:
      if (param):

         varNode[i]=(m.addVar(vtype=GRB.BINARY,name="node"+str(i), lb=0 ))

         m.update()
      else:
         varNode[i]=(m.addVar(vtype=GRB.CONTINUOUS,name="node"+str(i), lb=0 ))

         m.update()
   return (varEdges,varNode)




 
def objectif(var):
    obj = LinExpr();
    obj =0
    obj = quicksum([var[i][0]*var[i][1] for i in var])
    return obj
   

def contraintes(varEdges,varNode,m,k,listcycle=None) :
   
   #contraintes sur les cycles:
   if listcycle!=None:
      for cycle in listcycle:
         for i in range(-1,len(cycle)-1):
        
            xe=varEdges[(min(cycle[i],cycle[i+1]),max(cycle[i],cycle[i+1]))][0]

            liste=[]
            for j in range (0,-len(cycle)+1,-1):
               liste.append(-1*varEdges[(min(cycle[j+i],cycle[j+i-1]),max(cycle[j+i],cycle[j+i-1]))][0])
 

            m.addConstr(quicksum([xe]+liste ) <= 0)
          
   #contrainte 4
   for i in range(len(varNode)):
      for j in range(i+1,len(varNode)):
         if (i,j) in varEdges:
            m.addConstr(quicksum([varNode[j],-1*varEdges[(i,j)][0]] ) <= 0)
   #contrainte 5 
   
   
   for j in range(len(varNode)):
     liste=[]
     for i in range (0,j):
       if (i,j)in varEdges:
         #print(i,j)
         liste.append(-1*varEdges[(i,j)][0]   )
       else :
         liste.append(varNode[i])
     
     Dij=len(liste)
     m.addConstr(quicksum([varNode[j],Dij]+liste ) >= 1)
     #print(j,"+",Dij )    
   
   
   #contrainte 6
   
   m.addConstr(quicksum([varNode[j] for j in varNode] ) == k)

   #contrainte 7

   for j in range (len(varNode)):
      m.addConstr(varNode[j] >= 0)
      m.addConstr(varNode[j] <= 1)
      
   #contrainte 8
   for j in range(len(varNode)):
      for k in range (0,j):
         for i in range (0,k):
            
            if (i,j) in varEdges:
               if (k,j) in varEdges:
                  m.addConstr(quicksum([varNode[i],varNode[k],-1*varEdges[(i,j)][0],-1*varEdges[(k,j)][0]])<=1)
         
   return  



def constructGraphe0(Node,listEdge,listEdgeValue):
	G0=nx.Graph()
	G0.add_nodes_from([i for i in range(Node)])
	for i in range(len(listEdge)):

		if (int(listEdgeValue[i])==0):

			c=2
			while (listEdge[i].VarName[c]!='_'):
				c+=1

			G0.add_edge(int(listEdge[i].VarName[1:c]),int(listEdge[i].VarName[c+1:]))
		



	return G0

def callBack(model,where):	
	if where== GRB.Callback.MIPSOL:
		nodeValue=[x for x in model.getVars() if x.VarName.find('node') != -1]
		edgeVars=[x for x in model.getVars() if x.VarName.find('x') != -1]
		edgeValue=[model.cbGetSolution(x) for x in edgeVars]
		G0=constructGraphe0(len(nodeValue),edgeVars,edgeValue)		

		nodeValue1=[int(x.VarName[4:]) for x in nodeValue if (round(model.cbGetSolution(x))==1)]
		for i in range(0,len(nodeValue1)):
			for j in range(i+1,len(nodeValue1)):
				if nx.has_path(G0,nodeValue1[i],nodeValue1[j]):
					path=nx.shortest_path(G0,nodeValue1[i],nodeValue1[j],weight=None)
					liste=[]
					for p in range(0,len(path)-1):
						a=[x for x in model.getVars() if x.VarName.find('x'+str(path[p])+"_"+str(path[p+1])) != -1 or x.VarName.find('x'+str(path[p+1])+"_"+str(path[p])) != -1]
						liste.append(-1*a[0])
					node=[x for x in model.getVars() if x.VarName.find('node'+str(nodeValue1[i])) != -1 or x.VarName.find('node'+str(nodeValue1[j])) != -1 ]
					liste.append(node[0])
					liste.append(node[1])
					print("contrainte en plus 1")
					print(liste)
					model.cbLazy(quicksum(liste) <= 1)
		
		edgeValue1=[]
		for i in range(len(edgeVars)):

			if (int(edgeValue[i])==1):

				c=2
				while (edgeVars[i].VarName[c]!='_'):
					c+=1
				nodeSrc=int(edgeVars[i].VarName[1:c])
				nodeGoal=int(edgeVars[i].VarName[c+1:])
				if nx.has_path(G0,nodeSrc,nodeGoal):
					path=nx.shortest_path(G0,nodeSrc,nodeGoal,weight=None)
					liste=[]
					for p in range(0,len(path)-1):
						a=[x for x in model.getVars() if x.VarName.find('x'+str(path[p])+"_"+str(path[p+1])) != -1 or x.VarName.find('x'+str(path[p+1])+"_"+str(path[p])) != -1]
						liste.append(-1*a[0])
					liste.append(edgeVars[i])
					print("contrainte en plus 2")
					print(liste)
					model.cbLazy(quicksum(liste) <= 0)
    
    

		

					
	return
	
def sousEnsembleBaseCycle(G,L):
  
  n=G.number_of_nodes()
  base=nx.cycle_basis(G)
  base2=[]
  for x in base:
    if len(x)<=L:
      base2.append(x)#On obtient la base sans les cycle de longueur > L
  if len(base2)<n:
    return base2
  else:
    base3=[]
    indice=0
    while(len(base3)!=n):
      rand=rd.random()
      if rand<1.0/len(base2):
        base3.append(base2[indice])
      indice+=1
      if indice==len(base2):
        indice=0
  return base3
      
    

def resolutionplne1 (G,k,param=True):
   m = Model("mogplex")
   
   listcycle=nx.cycle_basis(G)
   
   #listcycle=sousEnsembleBaseCycle(G,L=G.number_of_nodes()/4)
   #variables
   print("yo",listcycle)
   varEdges,varNode=creervar(G,m,param=param)
   
   
   #fonction objectif
   obj=objectif(varEdges)   
   m.setObjective(obj,GRB.MINIMIZE)#on ajoute la fonction objectif au model
   
   #contraintes
   if (param):
      m.Params.lazyConstraints=1
      contraintes(varEdges,varNode,m,k)
      m.optimize(callBack)
      nodeValue=[x for x in m.getVars() if x.VarName.find('node') != -1]
      edgeVars=[x for x in m.getVars() if x.VarName.find('x') != -1]
      edgeValue=[x.x for x in edgeVars]
      print(len(edgeVars))
      print(len(edgeValue))
      #G0=constructGraphe0(len(nodeValue),edgeVars,edgeValue)
      #nx.draw(G0)	
      #plt.show()
   else:
      contraintes(varEdges,varNode,m,k,listcycle=listcycle)
      m.optimize()



   """
   for j in varNode:
      print(j)
      print(varNode[j].x)
   for j in varEdges:
      print(j)
      print(varEdges[j][0].x)"""
   return 0






#resolutionplne1(G,3,param=True)


def resolutionplne2(G,k,L,param=True):
   m = Model("mogplex")
   
   listcycle=nx.cycle_basis(G)
   
   for l in listcycle:
       if len(l)>L:
           listcycle.remove(l)
   
   #listcycle=sousEnsembleBaseCycle(G,L)
   #print(listcycle)
   #variables
   #print("yo",listcycle)
   varEdges,varNode=creervar(G,m,param=param)
   
   
   #fonction objectif
   obj=objectif(varEdges)   
   m.setObjective(obj,GRB.MINIMIZE)#on ajoute la fonction objectif au model
   
   #contraintes
   if (param):
      m.Params.lazyConstraints=1
      contraintes(varEdges,varNode,m,k)
      m.optimize(callBack)
      nodeValue=[x for x in m.getVars() if x.VarName.find('node') != -1]
      edgeVars=[x for x in m.getVars() if x.VarName.find('x') != -1]
      edgeValue=[x.x for x in edgeVars]
      #print(len(edgeVars))
      #print(len(edgeValue))
      #G0=constructGraphe0(len(nodeValue),edgeVars,edgeValue)
      #nx.draw(G0)	
      #plt.show()
   else:
      #print("hello")
      contraintes(varEdges,varNode,m,k,listcycle=listcycle)
      m.optimize()

   #print("hhh",m.get

   """
   for j in varNode:
      print(j)
      print(varNode[j].x)
   for j in varEdges:
      print(j)
      print(varEdges[j][0].x)"""
   return m.ObjVal

#G=graphe_creux_dens_weight2(50,0.1)
#G.edges(data=True)
#print(G.nodes(), G.edges())
#print(resolutionplne1(G,30,param=False))
#print(resolutionplne1(G,15,param=True))


def testTime(nbmin,nbmax,dens,L):
    t1=[]
    t2=[]
    for i in range(nbmin,nbmax):
        G=graphe_creux_dens_weight2(i,dens)
        T1=time.time()
        resolutionplne2(G,i/4,L,param=False)
        T2=time.time()
        t1.append(T2-T1)
        T1=time.time()
        #resolutionplne1(G,i/4,param=True)
        T2=time.time()
        t2.append(T2-T1)
    plt.plot(t1,"r")
    plt.plot(t2,"b")
    plt.show()
    return

#testTime(20,25,0.1,7)  

def testResult(n,dens,Lmin,Lmax):
    G=graphe_creux_dens_weight2(n,dens)
    liste=np.zeros(Lmax-Lmin)
    temps=np.zeros(Lmax-Lmin)
    for l in range(Lmin,Lmax):
        t1=time.time()
        liste[l-Lmin]+=resolutionplne2(G,n/4,l,param=False)
        t2=time.time()
        temps[l-Lmin]+=t2-t1
    #plt.plot(liste)
    #plt.show()
    return liste,temps
testResult(30,0.1,3,20)

def test2(n,dens,Lmin,Lmax,prec):
    liste=np.zeros(Lmax-Lmin)
    temps=np.zeros(Lmax-Lmin)
    for i in range( prec):
        
        l,t=testResult(n,dens,Lmin,Lmax)
        liste+=l
        temps+=t
    abscisse=[i for i in range(Lmin,Lmax)]
    plt.subplot(221)
    plt.plot(abscisse,liste/prec)
    plt.xlabel("L")
    plt.ylabel("résultat")
    plt.show()
    plt.subplot(222)
    plt.plot(abscisse,temps/prec)
    plt.xlabel("L")
    plt.ylabel("temps d'éxecution")
    plt.show()
    return
#test2(150,0.1,3,11,100)

def test3(nmin,nmax,Lmin,Lmax,dens,prec):
    listeR=np.zeros((Lmax-Lmin,nmax-nmin))
    listeT=np.zeros((Lmax-Lmin,nmax-nmin))
    for i in range(nmin,nmax):
        for j in range(prec):
            G=graphe_creux_dens_weight2(i,dens)
            for k in range(Lmin,Lmax):
                t1=time.time()
                listeR[k-Lmin,i-nmin]+=resolutionplne2(G,i/4,k,param=False)
                t2=time.time()
                listeT[k-Lmin,i-nmin]+=t2-t1
    colorNames = ["r","white","g","white","k","white","c"]
    abscisse=[i for i in range(nmin,nmax)]
    p1=[]
    plt.subplot(223)
    for k in range(Lmin,Lmax):
        p1.append(plt.plot(abscisse,listeR[k-Lmin]/prec,colorNames[k-Lmin]))
    
    plt.ylabel("résultat")
    plt.xlabel("nombre de noeuds")
    #plt.legend(p1,[str(k) for k in range(Lmin,Lmax)])
    plt.show()
    p2=[]
    plt.subplot(224)
    for k in range(Lmin,Lmax):
        p2.append(plt.plot(abscisse,listeT[k-Lmin]/prec,colorNames[k-Lmin]))
    #plt.legend(p2,[str(k) for k in range(Lmin,Lmax)])
    plt.ylabel("temps d'éxécution")
    plt.xlabel("nombre de noeuds")
    #print("yyyyyyyyyyyyyy",len(p1))
    return

