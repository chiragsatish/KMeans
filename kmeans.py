import numpy as np
import random
import collections
import sys
from sklearn import metrics
from statistics import mode

def norm(x):
    """
    >>> Function you should not touch
    """
    max_val = np.max(x, axis=0)
    x = x/max_val
    return x

def rand_center(data,k):
    """
    >>> Function you need to write
    >>> Return Euclidean distance between two points.
    """
    randlist=[]
    i=0
    while i<k:
        r = random.randint(0,149)
        if r not in randlist:
            randlist.append(r)
            i+=1
    print(randlist)
    arr=[]
    for i in range(0,len(randlist)):
        arr.append(list(data[randlist[i],:]))
    
    return np.array(arr)
    pass

def converged(centroids1, centroids2):
    """
    >>> Function you need to write
    >>> check whether centroids1==centroids2
    """  
    return np.array_equal(centroids1,centroids2)      
    pass

def update_centroids(data, centroids, k=3):
    """
    >>> Function you need to write
    >>> Assign each data point to its nearest centroid based on the Euclidean distance
    >>> Update the cluster centroid to the mean of all the points assigned to that cluster
    """
    label=[]
    cluster_points={}
    centroids_cpy = np.copy(centroids)
    for each in data:
        distance = sys.maxsize
        each_label = -1
        for i in range(0,len(centroids)):
            d = np.linalg.norm(each-centroids[i,:])
            if(d<distance):
                distance=d
                each_label=i
        label.append(each_label)
        if each_label in cluster_points:
            cluster_points[each_label].append(list(each))
        else:
            cluster_points[each_label] = []
            cluster_points[each_label].append(list(each))
    for each in cluster_points:
        centroids_cpy[each] = np.array(np.mean(cluster_points[each],axis=0))
    return centroids_cpy,label
    pass

def kmeans(data,k=3):
    """
    >>> Function you should not touch
    """
    # step 1:
    #print(data)
    centroids = rand_center(data,k)
    print("initial centroids")
    print(centroids)
    converge = False
    iterations=0
    while not converge:
        iterations+=1
        old_centroids = np.copy(centroids)
        # step 2 & 3
        centroids, label = update_centroids(data, old_centroids)
        # step 4
        converge = converged(old_centroids, centroids)
    print(">>> final centroids")
    print(centroids)
    print("iterations:",iterations)
    return centroids, np.array(label)

def evaluation_sse(data,centroids):
    sum=0
    for each in data:
        sum+=min(np.linalg.norm(each-centroids[0,:]),np.linalg.norm(each-centroids[1,:]),np.linalg.norm(each-centroids[2,:]))
    print("SSE:",sum)
    pass

def evaluation_gini(predict, ground_truth):
    """
    >>> use the ground truth to do majority vote to assign a flower type for each cluster
    >>> accordingly calculate the probability of missclassifiction and correct classification
    >>> finally, calculate gini using the calculated probabilities
    """
    print("GINI evaluation")
    gini_sum=0
    cluster_points={}
    assigned_label = [0,0,0]
    predict = list(predict)
    ground_truth = list(ground_truth)
    for i in range(0,150):
        if predict[i] in cluster_points:
            cluster_points[predict[i]].append(ground_truth[i])
        else:
            cluster_points[predict[i]] = list()
            cluster_points[predict[i]].append(ground_truth[i])
    for each in cluster_points:
        counter=collections.Counter(cluster_points[each])
        print(counter)
        if len(counter)>1:
            for label in counter:
                if counter[label]<=max(counter.values()):
                    gini_sum+=counter[label]
            gini_sum-=max(counter.values())
    gini_val = float(gini_sum)/150.0
    print("GINI:",gini_val)
    pass

