from scipy.linalg import svd
from scipy.linalg import eig
from numpy import *
import numpy as np
import pandas as pd
from cluster_louvain import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.utils.extmath import randomized_svd
import operator

import matplotlib.cm as cm
import networkx as nx
cln=6 #number of clusters, global variable

def cluster(preiden,iden):

    l=len(preiden)

    for i in range(l):
        t=len(preiden[i])
    
        degcount=np.zeros((cln))

        for j in range(t):
            t=round(preiden[i][j])
            t1=round(iden[t])
            degcount[t1]=degcount[t1]+1

        
        print(degcount.astype(int))

def printplotsave(printer,checker,nadj,n1):    

    print("We start nice graph")
    x=[]
    y=[]
    txt=[]

    l1=len(checker)

    for i in range(1,l1//2,100):
        x.append(printer[i])
        y.append(i)
        checker1=checker[0:i]
        avg1=mean_degree(checker1,n1,nadj)
        txt.append(avg1)

    plt.xlabel("Inner product with top eigenvector")
    plt.ylabel("The vertices in G'")
    plt.scatter(x, y)

    for i in range(len(txt)):
        plt.annotate(txt[i], (x[i], y[i]))


    print("We are at the end")    
    #ax.show() 
    #figure(figsize=(16, 6), dpi=80)
    #plt.plot(printer,label=1)
    plt.legend()  
    plt.savefig("sproj1.png")  
    plt.show()    

def printplot(printer,n1):    
    figure(figsize=(16, 6), dpi=80)
    plt.plot(printer,label=1)
    plt.legend()    
    plt.show()    

def mapping(sol,iden,n):

    cur=[]
    rcur=np.zeros((n))
    c=0
    for i in range(n):
        rcur[i]=-1

    for i in range(n):
        if(sol[i]==-1):
            t=iden[i]
            cur.append([i,t])
            rcur[i]=c
            c=c+1 
            
    return cur,rcur

def makegraph(sol,iden,n,per,scomp):
    cur,rcur=mapping(sol,iden,n)
    n1=len(cur)
    ty=np.shape(scomp)
    ll=ty[0]
    #ll=(n*(n-1))//2
    print("n1=",n1)


    inter=np.zeros((n1))
    intra=np.zeros((n1))
    nadj=np.zeros((n1,n1))
    tc=0
    tcc=0
    for i in range(round(ll*per)):
        tv1=sol[round(scomp[i][3])]
        tv2=sol[round(scomp[i][4])]
        if(tv1==-1 and tv2==-1):
            tc=tc+1
            v1=rcur[round(scomp[i][3])]
            v2=rcur[round(scomp[i][4])]
            if(v1!=scomp[i][3] or v2!=scomp[i][4]):
                tcc=tcc+1
            nadj[round(v1)][round(v2)]=1
            nadj[round(v2)][round(v1)]=1

            
    for i in range(n1):
        for j in range(i):
            t1=round(cur[i][1])
            t2=round(cur[j][1])
            if(t1==t2):
                intra[i]=intra[i]+nadj[i][j]
                intra[j]=intra[j]+nadj[i][j]                
            else:
                inter[i]=inter[i]+nadj[i][j]
                inter[j]=inter[j]+nadj[i][j]
                
            
    
            
    tdeg=nadj.sum(axis=1)
    rowmean=nadj.mean(axis=1)
    
    
    nadj1=np.zeros((n1,n1))
    for i in range(n1):
        nadj1[:,i]=nadj[:,i]-rowmean

    #gU,gs,gVT=svd(nadj1)

    gU,gs,gVT = randomized_svd(nadj1, n_components=20, n_iter=5, random_state=None)

    return n1,nadj,nadj1,cur,rcur,gU,tdeg

# Sort the nodes according to their inner product with the eigenvector
def pluralset(nadj1,n1,pcn,gU):

    proj=np.zeros((n1,2))

    for i in range(n1): #for each node
        proj[i][0]=np.dot(gU[:,pcn],nadj1[:,i]) #inner product:<largest eigenvector, node i>
        proj[i][1]=i #node i

    sproj=sorted(proj, key=operator.itemgetter(0), reverse=True) #sorted according to inner product value, proj[i][0]

    return sproj          

#Here we recover

def recover1(sproj,n1,tdeg,nadj,cutoff):

    print("Cutoff contains",cutoff,"many vertices")
    
    
    nv=np.zeros((n1,4))
    for i in range(n1):
        nv[i][0]=i
        for j in range(cutoff):
            nv[i][1]=nv[i][1]+nadj[i,round(sproj[j][1])]
        
        nv[i][2]=tdeg[i]-nv[i][1]

    mult=((n1-cutoff)/cutoff)
    for i in range(n1):
        if(nv[i][2]==0):
            nv[i][3]=mult*nv[i][1]
        else:
            nv[i][3]=mult*nv[i][1]/nv[i][2]

    snv=sorted(nv, key=operator.itemgetter(1), reverse=True)
            
    return nv,snv,n1

def recover11(rset,n1,tdeg,nadj,cutoff):

    print("Welcome to repeated recovery")
    print("Cutoff contains",len(rset),"many vertices")
    
    
    nv=np.zeros((n1,4))
    for i in range(n1):
        nv[i][0]=i
        for j in range(len(rset)):
            nv[i][1]=nv[i][1]+nadj[i][round(rset[j])]
        
        nv[i][2]=tdeg[i]-nv[i][1]

        
        
    mult=((n1-cutoff)/cutoff)
    for i in range(n1):
        if(nv[i][2]==0):
            nv[i][3]=mult*nv[i][1]
        else:
            nv[i][3]=mult*nv[i][1]/nv[i][2]

    snv=sorted(nv, key=operator.itemgetter(1), reverse=True)
            
    return nv,snv,n1

def recfilter(tdeg,track,nadj,n1):

    cutoff=len(track)
    nv=np.zeros((n1,3))
    for i in range(n1):
        nv[i][0]=i
        for j in range(cutoff):
            nv[i][1]=nv[i][1]+nadj[i,round(track[j])]
        
        nv[i][2]=tdeg[i]-nv[i][1]


    snv=sorted(nv, key=operator.itemgetter(1), reverse=True)
            
    return nv,snv,n1

def filterset(track,nadj,n1):

    ll=len(track)
    deg=np.zeros((ll))

    for i in range(ll):
        
        t11=round(track[i])
        for j in range(ll):

            t12=round(track[j])
            deg[i]=deg[i]+nadj[t11][t12]

    avg=mean(deg)

    clean=[]

    for i in range(ll):
        if(deg[i]>(avg/1.5)):
            clean.append(track[i])


    return clean

def clusterid(track,cur):

    degcount=np.zeros((cln))
    ll=len(track)

    for i in range(ll):
        t=round(cur[round(track[i])][1])
        degcount[t]=degcount[t]+1

    return degcount
def majorityid(rset,cur):
    degcount=clusterid(rset,cur)

    mx=0
    pos=-1
    for i in range(cln):
        if(degcount[i]>mx):
            mx=degcount[i]
            pos=i

    return pos

def mean_degree(node_list,n1,nadj):

    l1=len(node_list)
    deg=np.zeros((l1))
    for i in range(l1):
        for j in range(l1):
            t11=round(node_list[i])
            t12=round(node_list[j])
            deg[i]=deg[i]+nadj[t11][t12]
    
    avg=mean(deg)
    return avg
#Recovery of cluster

def recover2(snv,n1,recpar,sol,current,cur,bl):

       
    print("Recovery step")    

    for i in range(recpar):
        t1=round(snv[i][0])
        t2=round(cur[t1][0])
        sol[t2]=current

    t=0
    for i in range(recpar,(recpar+bl)):
        t1=round(snv[i][0])
        t2=round(cur[t1][0])
        sol[t2]=-5
        t=t+1

    print("At recovery, removed",t)
    current=current+1    
    return current

def recovernew(buffers,n1,sol,current,cur):

    print("Recovery step")  
    l=len(buffers)   

    for i in range(l):
        t1=round(buffers[i])
        t2=round(cur[t1][0])
        sol[t2]=current

    current=current+1    
    return current

#create the lists

def clusterify(sol, current, n):

    preiden=[]
    remain=[]
    for i in range(current):
         preiden.append([])


    for i in range(n):
        if(sol[i]!=-1):
            t=round(sol[i])
            preiden[t].append(i)
        else:
            remain.append(i)

    return preiden,remain

def seuratcluster(seurat):

    l1=len(seurat)
    mx=0
    for i in range(l1):
        if(seurat[i]>mx):
            mx=seurat[i]

    l2=mx+1


    seuratiden=[]
    for i in range(l2):
        seuratiden.append([])

    for i in range(l1):
        t=round(seurat[i])
        seuratiden[t].append(i)

    return seuratiden