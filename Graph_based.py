import pandas as pd
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import log2 


def entropy(p): 
    ent = p * log2(p)
    return -ent

def sub_matrix(index,item_matrix):
    return item_matrix[index,:]


def filter_top_items(item_matrix,target_N,user_index):
    occurance = item_matrix.sum(axis = 0)
    topN = occurance.argsort()[-2*(target_N):][::-1]
    for i in range(0,len(user_index)):
        if(np.nonzero(topN == user_index[i])[0].size != 0):
            topN = np.delete(topN,np.nonzero(topN == user_index[i])[0][0])
    
    while(topN.size > target_N):
        topN = np.delete(topN,target_N)

    return topN



#data = pd.read_excel(r'C:\Users\H.HUANG\Desktop\CS4195-Modeling-Networks-Project-lhawx0-patch-1\item.xlsx')
#matrix = data.to_numpy()
matrix = np.load('itemmatrix.npy')
matrix = np.delete(matrix,0,1)
user_index = [1,2,3,4,5,6,7,173,93,49]
co_rated = sub_matrix(user_index,matrix)
top_set = filter_top_items(co_rated,10,user_index)

input()


