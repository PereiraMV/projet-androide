# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:34:30 2018

@author: lucas
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rd

"""
g=nx.Graph()
g.add_nodes_from([i for i in range(100)])
print(g.nodes())


"""
def graphe_creux(n):
    G=nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    liste_noeuds=[i for i in range(1,n)]
    L=np.ones(n)
    L=L/(n*1.0)
    #probabilité de creer une arrete avec le i-ieme noeud
    current=0
    #tant que le graphe n'est pas connexe
    while len(liste_noeuds)>0:
        rand=rd.random()
        s=0
        for k in range(n):
            
            if rand>=s and rand<=s+L[k]:
                G.add_edge(current,k)
                current=k
                if k in liste_noeuds:
                    liste_noeuds.remove(k)
                    #k est dans la partie connexe du graphe maintenant
                norm=1-(L[k]/2.0)
                L[k]/=2.0#la proba de se lier à k baisse
                L=L/(norm)#celle des autres noeuds augmente
                break
            s+=L[k]
    #print(np.sum(L))
    return G

def graphe_creux_dens(n,dens):
    G=nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    liste_noeuds=[i for i in range(1,n)]
    L=np.ones(n)
    L=L/(n*1.0)
    d=0
    n2=1/n**2
    #probabilité de creer une arrete avec le i-ieme noeud
    current=0
    #tant que le graphe n'est pas connexe
    while (len(liste_noeuds)>0 or d<dens):
        rand=rd.random()
        s=0
        for k in range(n):
            
            if rand>=s and rand<=s+L[k]:
                if current!=k:    
                    G.add_edge(current,k)
                    current=k
                if k in liste_noeuds:
                    liste_noeuds.remove(k)
                    #k est dans la partie connexe du graphe maintenant
                norm=1-(L[k]/2.0)
                L[k]/=2.0#la proba de se lier à k baisse
                L=L/(norm)#celle des autres noeuds augmente
                break
            s+=L[k]
        d+=n2
    #print(np.sum(L))
    return G
G=graphe_creux(20)
#G=graphe_creux_dens(20,0.1)

#print(len(G.edges()))
#print(len(G2.edges()))

#print(nx.cycle_basis(G))
"""
print(G.nodes())
print(G.edges())
print(G.edges(1)[0])
print(G.edges(1)[1][1])



"""

def testCycle(n,dens,nb):
    s=0.
    for i in range(nb):
        
        G=graphe_creux_dens(n,dens)
        l=nx.cycle_basis(G)
        s+=np.array([len(k) for k in l]).mean()
    return s/nb

def printTCycle(nmin,nmax,dens,prec):
    l=np.zeros(nmax-nmin)
    for i in range(nmin,nmax):
        l[i-nmin]=testCycle(i,dens,prec)
    plt.plot(l)
    plt.show()

#printTCycle(20,100,0.1,10)

#print(testCycle(10,0.1,5))



def graphe_creux_dens_weight(n,dens):
    G=nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    liste_noeuds=[i for i in range(1,n)]
    L=np.ones(n)
    L=L/(n*1.0)
    d=0
    n2=1./n**2
    arretes=[]
    cpt=0
    
    #probabilité de creer une arrete avec le i-ieme noeud
    current=0
    #tant que le graphe n'est pas connexe
    while (len(liste_noeuds)>0 or d<dens):
        rand=rd.random()
        s=0
        cpt+=1
        for k in range(n):
            
            if rand>=s and rand<=s+L[k]:
                arretes.append([current,k,0])
                current=k
                if k in liste_noeuds:
                    liste_noeuds.remove(k)
                    #k est dans la partie connexe du graphe maintenant
                norm=1-(L[k]/2.0)
                L[k]/=2.0#la proba de se lier à k baisse
                L=L/(norm)#celle des autres noeuds augmente
                break
            s+=L[k]
        d+=n2
    weight=[1 for i in range(int(cpt/2)) ]+[-1 for i in range(int(cpt/2)+1)]
    #print(weight)
    rd.shuffle(weight)
    #print(weight,len(weight),sum(weight))
    
    
    for i in range(len(arretes)):
        arretes[i][2]=weight[i]
        G.add_edge(arretes[i][0],arretes[i][1],weight=arretes[i][2])
        
    #G.add_edges_from(arretes)
    #G.add_weighted_edges_from(arretes)
    #print(G.edges(),arretes)
    #print(np.sum(L))
    return G


def graphe_creux_dens_weight2(n,dens):
    G=nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    liste_noeuds=[i for i in range(1,n)]
    L=np.ones(n)
    L=L/(n*1.0)
    d=0
    n2=1.0/n**2
    arretes=[]
    cpt=0
    
    #probabilité de creer une arrete avec le i-ieme noeud
    current=0
    #tant que le graphe n'est pas connexe
    while (len(liste_noeuds)>0 or d<dens):
        rand=rd.random()
        s=0
        cpt+=1
        for k in range(n):
            
            if rand>=s and rand<=s+L[k]:
                if current!=k:
                    arretes.append([current,k,0])
                    current=k
                if k in liste_noeuds:
                    liste_noeuds.remove(k)
                    #k est dans la partie connexe du graphe maintenant
                norm=1-(L[k]/2.0)
                L[k]/=2.0#la proba de se lier à k baisse
                L=L/(norm)#celle des autres noeuds augmente
                break
            s+=L[k]
        d+=n2
    #weight=[int(rd.random()*10**5) for i in range(cpt) ]
    weight=np.random.normal(10,0.5,cpt)*(10**5)
    weight=weight.astype(int)
    #print("yo",weight)
    rd.shuffle(weight)
    #print(weight,len(weight),sum(weight))
    
    
    for i in range(len(arretes)):
        arretes[i][2]=weight[i]
        G.add_edge(arretes[i][0],arretes[i][1],weight=arretes[i][2])
        
    #G.add_edges_from(arretes)
    #G.add_weighted_edges_from(arretes)
    #print(G.edges(),arretes)
    #print(np.sum(L))
    return G







#G=graphe_creux_dens_weight2(10,0.2)
#G.add_edge(1,2,weight=1)

#print("aa",G.edges())




def testNbArretes(int_min,int_max,prec,dens):
    
    liste=np.zeros((int_max-int_min))
    for i in range(int_min,int_max):
        #print(i)
        for k in range(prec):
            g=graphe_creux_dens(i,dens)
            
            liste[i-int_min]+=len(g.edges())
        
    liste/=prec
    return liste


#print(testNbArretes(400,410,3))   

def afficherTest(int_min,int_max,prec,dens):
    #x=np.linspace(int_min,int_max)
    plt.plot(testNbArretes(int_min,int_max,prec,dens))
    plt.show
        
#afficherTest(100,350,5,0.05)
#pos=nx.fruchterman_reingold_layout(G)
#pos=nx.spring_layout(G)
#nx.draw(G,with_labels=True,font_weight='bold')
"""
G=nx.Graph()

G.add_edge('a','b',weight=0.6)
G.add_edge('a','c',weight=0.2)
G.add_edge('c','d',weight=0.1)
G.add_edge('c','e',weight=0.7)
G.add_edge('c','f',weight=0.9)
G.add_edge('a','d',weight=0.3)
#print(G.edges())
"""
def afficher_graph_dens_weight(n,dens):
    G=graphe_creux_dens_weight2(n,dens)
    #print(G.edges())
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <10**6+10000 and d['weight'] >10**6-10000]
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >10**6+10000 or d['weight'] <10**6-10000]
    #erand=[(u,v) for (u,v,d) in G.edges(data=True) if abs(d['weight']) !=1]
    pos=nx.fruchterman_reingold_layout(G)
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,node_size=700)

    # edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge,
                           width=6)
    nx.draw_networkx_edges(G,pos,edgelist=esmall,
                           width=6,alpha=0.5,edge_color='b',style='dashed')
    #nx.draw_networkx_edges(G,pos,edgelist=erand,
                           #width=6,alpha=0.5,edge_color='r',style='dashed')
    
    # labels
    nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
    
    plt.axis('off')
    #plt.savefig("weighted_graph.png") # save as png
    plt.show() # display

#print(len(G.edges()))
#plt.show()

#afficher_graph_dens_weight(10,0.1)



def cycle_L(G,L):
    base=nx.cycle_basis(G)
    base2=[]
    #print(base)
    for x in base:
        #print(len(x))
        if len(x)<=L:
            base2.append(x)
            #print('yo')
    return base2



#print(cycle_L(G,5))


def cycle_vect(G,L):
    arretes=G.edges()
    #print(arretes)
    cycle_base=cycle_L(G,L)
    base_vect=np.zeros((len(cycle_base),len(arretes)))
    for x in cycle_base:
        
        for i in range(-1,len(x)-1):
            if x[i]<x[i+1]:
                ind2=arretes.index((x[i],x[i+1]))
            else:
                ind2=arretes.index((x[i+1],x[i]))
            ind1=cycle_base.index(x)
            base_vect[ind1,ind2]=1
    return base_vect

#print(cycle_vect(G,5))
            











  
