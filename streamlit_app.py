#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import statistics as s

#st.set_page_config(layout="wide")

silos=9
n_clusters=2
#metric=c1.selectbox("metric",["Idade Materna","Bishop Score","Cesarianas Anterior","Cesarianas"])

means={}
means["Idade Materna"]=[30.94473361910594, 30.620558542021765, 31.077226489516296, 31.091688089117394, 31.377103122865833, 31.31202023726448, 31.292021688613477, 31.35806504330773, 30.137582625118036]
means["Bishop Score"]=[5.654205607476635, 4.8772040302267, 5.408, 6.2594936708860756, 6.495614035087719, 5.5227272727272725, 5.826347305389222, 5.68, 6.042910447761194]
means["Cesarianas Anterior"]=[1.11864406779661, 0.5793376173999011, 1.1185647425897036, 1.1300813008130082, 0.31453804347826086, 0.5736070381231672, 0.6453608247422681, 0.8116646415552855, 0.7654205607476635]
means["Cesarianas"]=[0.3000612369871402, 0.2559328700668677, 0.24185177496367033, 0.22922022279348758, 0.27533804738866147, 0.29684228890439635, 0.2973147430932094, 0.27259356103938553, 0.22455146364494807]

st.markdown("""Please Select metric to assess in a distributed manner. Yor data will not be shared and only metadata will be collected from peers.""")


def calculate_centroids(seeds,mean,clusters):
    d=seeds.flatten()
    d=np.append(d,mean)
   # print(d)
    res=KMeans(n_clusters=clusters, random_state=0).fit(d.reshape(-1, 1))
    return res
    
def convergence_clusters_2(mean,clusters):
    new_seeds=np.zeros((silos,n_clusters))
    #get initial from all of the rest:
    c1_l=[]
    c2_l=[]
   # n = s.NormalDist(mu=50, sigma=10)
   # seeds = np.array(n.samples(n_clusters, seed=42))  # remove seed if desired
    seeds=np.array([np.random.randint(limit_rand, size=n_clusters) for i in range(silos)]) #mimic
    for i in range(10): #arbitrary, until convergence
        c1_=[]
        c2_=[]

        #create mine
        my_centroids=calculate_centroids(seeds,mean,n_clusters)
        #my_centroids=KMeans(n_clusters=clusters, random_state=0).fit(data.reshape(-1, 1))
      #  print(my_centroids.cluster_centers_)
        #get all the others
        for idx,x in enumerate(means[metric]):
            #row_no_null=x[~pd.isnull(x["IDADE_MATERNA"])]["IDADE_MATERNA"]
            silo_mean=x
            #means.append(silo_mean)
           # silo_own=KMeans(n_clusters=clusters, random_state=0).fit(row_no_null.values.reshape(-1, 1))
           # print(silo_own.cluster_centers_[:,0])
           # print(silo_mean)
            #silo_centroids=calculate_centroids(seeds,silo_own.cluster_centers_[:,0],n_clusters)
            silo_centroids=calculate_centroids(seeds,silo_mean,n_clusters).cluster_centers_

           # print(silo_centroids[:,0])
            new_seeds[idx,:]=silo_centroids[:,0]
            #print(new_seeds)
            c1_.append(silo_centroids.min())
            #print(silo_centroids.max())
            c2_.append(silo_centroids.max())

        seeds=new_seeds
        c1_l.append(np.mean(c1_))
        c2_l.append(np.mean(c2_))
      #  print(seeds)
    return c1_l,c2_l,seeds,means,my_centroids

def process_data(mean):
    print(mean)
    seeds=np.array([np.random.randint(100, size=n_clusters) for i in range(silos)])
       
        
    _,_,seed,means,my_centroids=convergence_clusters_2(mean,n_clusters)
  #  print(my_centroids.cluster_centers_[:,0])
    c1=plt.scatter([0],my_centroids.cluster_centers_[0,0])
    c2=plt.scatter([0],my_centroids.cluster_centers_[1,0])
    c3=plt.scatter([0],mean)
    plt.legend((c1, c2, c3),
               ('Cluster1', 'Cluster2', 'Means'),
               scatterpoints=1,
               loc=0,
               ncol=3,
               fontsize=8)
    plt.title(metric)

    st.pyplot(plt)       

c1,c2,c3=st.columns(3)
metric=c1.selectbox("metric",["Idade Materna","Bishop Score","Cesarianas Anterior","Cesarianas"])
mean=c2.number_input("Mean",min_value=0,value=0,step=0.1)
limit_rand=c3.number_input("Limit for Random",min_value=0,max_value=1000,value=100)

if st.button("Calculate"):
    process_data(mean)
    