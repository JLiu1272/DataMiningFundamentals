#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:23:33 2019

@author: JenniferLiu, JoeGolden
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.cluster import hierarchy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import copy

# Dendogram 
from scipy.cluster.hierarchy import dendrogram, linkage

class ShopperCluster(object):
    '''
    Object representing a shopper (or cluster of shoppers).
    '''

    def __init__(self, data):
        '''
        Initiates a cluster with a single shopper.
        :param data: a row of the csv INCLUDING ID, as a list
        '''
        #Ids of each shopper in the cluster
        self.ids = [data[0]]

        #The id for the entire cluster is simply the id of the first shopper added to the cluster
        self.id = data[0]

        #List of lists of all data besides id for each shopper (might not be necessary to hold onto, maybe delete later)
        self.data = [data[1:]]

        #The center of the cluster (starts as just the data of the initial shopper)
        #Do not access directly; use the get function to make sure it is updated
        self._center = data[1:]

        #Flag to make sure new centers are calculated before getting
        self.center_calculated = True

    def add_shopper(self, data):
        '''
        Add one shopper to the cluster. Sets the flag center_calculated to false so we know we haven't recalculated
        the center yet.
        :param data: a row of the csv INCLUDING ID, as a list
        '''

        self.ids.append(data[0])
        self.data.append(data[1:])
        self.center_calculated = False

    def add_two_shoppers(self, shopper):
        '''
        Merge two shoppers into one
        '''
        self.ids.extend(shopper.ids)
        self.data.extend(shopper.data)
        self.center_calculated = False 
        
        return self
    
    def get_center(self):
        '''
        Returns the center of the cluster.
        '''
        if not self.center_calculated:
            raise Exception('Attempted to access center of cluster before recalculating it.')
        return self._center

    def calculate_center(self):
        '''
        Recalculates and stores the center of the cluster.
        '''
        if len(self.ids) != len(self.data):
            raise Exception('Cluster ids list different length than data list - something went wrong adding an item.')

        total_items = len(self.ids)

        for index in range(len(self._center)):
            self._center[index] = sum([item[index] for item in self.data])/total_items

        self.center_calculated = True

class Cluster(object):
    def __init__(self, ids, centroid):
        
        # List of ids that are part of this cluster
        self.ids = ids
        
        # the centroid of this cluster 
        self.centroid = centroid 
    
    def get_ids(self):
        '''
        A getter function to get ids
        '''
        return self.ids
    
    def get_centroid(self):
        '''
        A getter function for retrieving centroid
        '''
        return self.centroid
    
    def update_centroid(self, new_centroid):
        '''
        Function to update the centroid
        '''
        self.centroid = new_centroid
    
class Agglomeration(object):
    def __init__(self, csv):
        
        # A mtx representing distances
        # between 2 clusters/shoppers
        self.dist_mtx = None
        
        # Load CSV 
        self.shoppers = pd.read_csv(csv)
        
        # List of all clusters/shoppers
        #Initiate by creating a cluster object for each individual shopper in order
        self.clusters = [ShopperCluster(list(data)) for index, data in self.shoppers.iterrows()]
        
        # Init heap 
        self.heap = []
        
        # a list of points that not considered 
        self.list_not_considered = []
        
    def init_pair_distance(self, clusters, heap):
        '''
        Initialise the pair-wise distances 
        
        Data structure of one single instance in heap 
        [distance, [clusterOuter, clusterInner]]
        '''
        
        # num clusters
        num_clusters = len(clusters)
        
        # Compute all pair
        # Compute the distances from point-n to every other
        # point in the data set 
        for outer in range(num_clusters):
            for inner in range(outer+1, num_clusters):
                
                # Extract cluster 1
                clusterOuter = clusters[outer]
                
                # Extract cluster 2
                clusterInner = clusters[inner]
                
                # Compute the distance
                dist = distance.euclidean(clusterOuter.get_center(), clusterInner.get_center())
                
                # Create a heap data structure 
                heap_item = [dist, [clusterOuter, clusterInner]]
                heap.append(heap_item)
        
        heap = sorted(heap, key=lambda tup: tup[0])
        
        return heap
    
    def merge_clusters(self, clusterOuter, clusterInner):
        '''
        Given two clusters. Merge them, and recalculate the center 
        '''
        
        # Create 1 shopper instance
        # with all the data values appended 
        new_cluster = clusterOuter.add_two_shoppers(clusterInner) 
        new_cluster.calculate_center()
        return new_cluster          
    
    def recompute_dist(self, heap, merged_cluster, prev_clusterInner):
        '''
        Since clusterOuter and clusterInner are now merged, 
        we need to remove any distances that stem from clusterOuter and
        clusterInner
        '''
        
        # When we merge clusters, we have to remove all the 
        # distance pairs that have 1 of the merged values.
        # this index keeps track of the clusters to be remove
        # to avoid redundacy 
        pop_idx = []
        
        prev_clusterIds = prev_clusterInner.ids
        #print(prev_clusterIds)
        
        for idx, item in enumerate(heap):
            clusterInner = item[1][0]
            clusterOuter = item[1][1]
            
            # if we find a distance pair entry where
            # one of the pairs is the merge cluster
            # we update their distance 
            # based on the cluster center 
            if clusterInner == merged_cluster:
                dist = distance.euclidean(clusterOuter.get_center(), merged_cluster.get_center())
                heap_item = [dist, [merged_cluster, clusterOuter]]
                heap[idx] = heap_item
            # It is possible that the distance pair comes from
            # the 2nd pair or 1st pair. So we check for both pairs and see
            # if any of them matches merged cluster
            elif clusterOuter == merged_cluster:
                dist = distance.euclidean(merged_cluster.get_center(), clusterInner.get_center())
                heap_item = [dist, [clusterInner, merged_cluster]]
                heap[idx] = heap_item
            
            # If the list of clusters equates to a previous cluster, we need to
            # remove it 
            if clusterOuter.ids == prev_clusterIds or clusterInner.ids == prev_clusterIds:
                pop_idx.append(idx)
        
        # Pop away all the indexes that need to be removed 
        heap = np.delete(np.array(heap), pop_idx, axis=0)
        
        # Sort the clusters 
        heap = sorted(heap, key=lambda tup: tup[0])
        
        return heap
    
    def print_heap(self, heap):
        for item in heap:
            print("Distance: {}".format(item[0]))
            print("Cluster Outer:", end='')
            print(item[1][0].ids)
            print("Cluster Inner:", end='')
            print(item[1][1].ids)
            print()
    
    def print_finalclusters(self, clusters):
        for idx,cluster in enumerate(clusters):
            print("Cluster - {}".format(idx))
            print("Distance: {}".format(cluster[0]))
            print("Cluster Outer:", end='')
            print(cluster[1][0].ids)
            print("Cluster Inner:", end='')
            print(cluster[1][1].ids)
            print("Merged Cluster:", end='')
            print(cluster[2])
            print()
    
    def remove(self, not_considered, clusterIds1, clusterIds2):
        '''
        Remove tuple in the not_considered list that
        contains the clusterIds
        '''
        new_not_considered = []
        
        # Any cluster with ids equivalent to clusterIds1 and clusterIds2 
        # are not considered 
        # It stupidly creates a new list
        for clust in not_considered:
            if clust[0] != clusterIds1 and clust[0] != clusterIds2:
                new_not_considered.append(clust)
        
        return new_not_considered 
    
    def update_not_considered(self, not_considered, clusterOuter, clusterInner, mergedCentroid):
        '''
        Remove the clusters that have been considered in the updated list.
        Return the newly updated list 
        '''
        
        # Remove the inner and outer clusters
        # create a merged inner and outer cluster and add to
        # list
        not_considered = self.remove(not_considered, clusterOuter.ids, clusterInner.ids)
        
        return not_considered
        
    
    def init_not_considered(self, clusters):
        '''
        Initially, all the clusters are its own node.
        Objects in not_considered are this format:
            1) [[cluster ids], [centroid of this cluster]]
        '''
        
        not_considered = []
        
        for clust in clusters:
            not_considered.append([clust.ids, clust.get_center()])
        
        return not_considered
    
    def label_points(self, not_considered):
        '''
        Convert the points into 1-5 values. 1 means this data point is
        in cluster 1. 5 means data point is in cluster 5 etc..
        '''
        
        # Initialise all shopper with a label of 0 
        shoppers = [0]*len(self.clusters)
        label_idx = 0
        
        # Sort the clusters based on size of cluster from largest cluster size to 
        # smallest cluster size 
        not_considered = sorted(not_considered, key=lambda tup: len(tup[0]), reverse=True)
        
        # Traverse through the 6 diff cluster
        for cluster, centroid in not_considered:
            # For every cluster, label it with the respective label 
            for shopper in cluster:
                shoppers[shopper-1] = label_idx
            
            label_idx += 1
        
        return shoppers 
    
    def print_centroids(self, not_considered):
        '''
        Nicely formats the centroid and what cluster this
        centroid is associated with
        '''
        
        idx = 1
        
        # Sort the clusters based on size of cluster from largest cluster size to 
        # smallest cluster size 
        not_considered = sorted(not_considered, key=lambda tup: len(tup[0]), reverse=True)
        
        print("**** Final 6 Clusters Information *****")
        for ids, centroid in not_considered:
            print("Cluster - {}".format(idx))
            print("Cluster Ids: ", end='')
            print(ids)
            print("Cluster Size: ", end='')
            print(len(ids))
            print("Cluster Centroid: ", end='')
            print(list(np.around(centroid, decimals=1)))
            print()
            idx += 1
        
    def cluster(self):
        '''
        The main function for clustering
        '''
        
        # num clusters 
        cluster_iter = len(self.clusters)
        
        # init pair distances 
        heap = self.init_pair_distance(self.clusters, self.heap)
        
        # Last clusters
        last_clusters = []
        
        # init list not considered
        # The point of this is keep track of the clusters 
        # We do not want clusters within clusters. Hence
        # we need to have separate lists for each different cluster 
        not_considered = self.init_not_considered(self.clusters)
        
        # When the size of not_considered is 6, we stop clustering 
        while len(not_considered) > 6:
            #print("****** Iter: " + str(cluster_iter) + " **********")
            #print(not_considered)
            
            # Find the minimum distance between centroids 
            min_distance_cluster = heap.pop(0)
            cluster_info = min_distance_cluster[1]
            clusterOuter = cluster_info[0]
            clusterInner = cluster_info[1]
            
            # The clusterOuter is being modified due to the 
            # merge cluster. Therefore, i need to create a deep
            # copy to prevent that from happening
            clusterOuterCpy = copy.deepcopy(clusterOuter)
            clusterInnerCpy = copy.deepcopy(clusterInner)
            
            # Create a new shopper list, where the two clusters are merged 
            merged_cluster = self.merge_clusters(clusterOuter, clusterInner)
            
            # Grab centroid of newly merged cluster
            mergedCentroid = merged_cluster.get_center()
            
            # Update the not considered list. Remove the old clusters and create
            # a new one with merged cluster and note its centroid
            not_considered = self.update_not_considered(not_considered, clusterOuterCpy, clusterInnerCpy, mergedCentroid)
            
            heap = self.recompute_dist(heap, merged_cluster, clusterInner)
            
            #self.print_heap(heap)
            
            cluster_iter -= 1
        
        # Create 1-5 labels, so that plotting can 
        # know which cluster associate to which color 
        shoppers = self.label_points(not_considered)
        
        # Print last 20 merges 
        print([len(shopper[0]) for shopper in not_considered])
        
        # Print cluster centroid info 
        self.print_centroids(not_considered)
        
        return shoppers

def find_bestAttribute(eig_vals, eig_vecs):
    '''
    Find the maximum eigen value and its corresponding
    eigen vector 
    '''
    max_ids = []
    max_eigens =[]
    # Create copy so shallow copying does not occur 
    eig_vals_dup = copy.deepcopy(eig_vals)
    
    
    for idx in range(4):
        max_id = np.argmax(eig_vals_dup)
        max_ids.append(max_id)
        max_eigens.append(eig_vals[max_id])
        
        eig_vals_dup[max_id] = False
    
    return max_ids, max_eigens

def find_worseAttribute(eig_vals, eig_vecs):
    '''
    Find the minimum eigen value and its corresponding
    eigen vector 
    '''
    min_ids = []
    min_eigens =[]
    eig_vals_dup = copy.deepcopy(eig_vals)
    
    # Find the max eigenvalue and remove it
    # from list
    # do this 4 times, sot hat we can find the 4
    # top eigenvalues 
    for idx in range(4):
        min_id = np.argmin(eig_vals_dup)
        min_ids.append(min_id)
        min_eigens.append(eig_vals[min_id])
        
        eig_vals_dup[min_id] = float("inf")
    
    return min_ids, min_eigens

def main():
    """
    Function that calls function in
    the correct sequence
    """

    csv = "HW_PCA_SHOPPING_CART_v850_KEY.csv"

    #Get dataframe from csv
    shoppers = pd.read_csv(csv)
    
    
    #Have to fix indices because pandas screws them up
    shoppers.columns = shoppers.columns.to_series().apply(lambda x: x.strip())
    
    attribute_labels = shoppers.columns[1:]
    
    raw_data = shoppers.drop(["ID"], axis=1)

    # Create labels for Dendogram
    labellst = [(shoppers["ID"][idx]) for idx in range(len(raw_data))]

    # Adjust size of dendo plot
    graph = plt.figure(figsize=(30, 30))

    # Create the dendogram using linkage
    Z = linkage(raw_data, method='centroid', metric='euclidean')
    #agg_clusters = hierarchy.fcluster(Z, 10, criterion='distance', R=None, monocrit=None)
    dendrogram(Z, leaf_rotation=0, orientation='right', leaf_font_size=8, labels=labellst)
    plt.show()
    
    # Agglomeration clustering process
    agg = Agglomeration(csv)
    agg_clusters = agg.cluster()
    
    #("Clusters: ")
    #print(agg_clusters[:10])

    covariance_matrix = np.cov(raw_data.T)

    #Get eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)

    #Convert eigenvalues and vectors to real numbers
    #(They originally return as complex numbers due to small differences caused by machine precision errors,
    #however from what I have read this means the complex parts of the values are insignificant and can just be removed)
    eig_vals, eig_vecs = np.abs(np.real(eig_vals)), np.real(eig_vecs)
    

    # Change the float format to only display 1 significant digit 
    pd.options.display.float_format = '{:.1f}'.format
    
    # print all the eigen values
    print("All Eigenvalues:")
    print(np.array(np.real(eig_vals)))
    print()
    
    eigen_pd = pd.DataFrame(eig_vecs, columns=attribute_labels)
    print(eigen_pd.head(20))
    #print([eigen_pd == '')
    
    #Print out the top 4 eigenvalues
    #np.set_printoptions(precision=2)
    print("Top 4 Eigenvalues: ")
    #print(np.array(np.real(eig_vals[:4])))
    max_ids, max_eigens = find_bestAttribute(eig_vals, eig_vecs)
    print(max_eigens)
    
    print("Most Important Attributes:")
    print()
    
    num_eigen = 1
    top_eigenvectors = []
    for idx in max_ids:
        print("Eigenvector " + str(num_eigen) + ":", end=' ')
        print(list(np.abs(np.around(eigen_pd.iloc[idx], decimals=1))))
        print()
        top_eigenvectors.append(list(np.abs(np.around(eigen_pd.iloc[idx], decimals=1))))
        num_eigen += 1
    
    summation = np.around(np.sum(top_eigenvectors, axis=0), decimals=1)
    print("Total :", end=' ')
    print(list(summation))
    
    #Now, take the dot product to project onto the first, second, and fourth eigenvectors
    proj = np.dot(raw_data, np.vstack([eig_vecs[:2], eig_vecs[3]]).T)

    agglom_3d = plt.figure()
    ax = agglom_3d.add_subplot(111, projection='3d')

    #Set up colors by agglom clusters
    LABEL_COLOR_MAP = {0: 'c', 1: 'r', 2: 'b', 3: 'g', 4: 'm', 5: 'black'}
    label_color = [LABEL_COLOR_MAP[l] for l in agg_clusters]

    ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=label_color, marker='o')

    plt.show()

    # rotate the axes and update
    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(.001)

    #Now let's perform k-means on this data.
    #We're gonna try for 5 clusters since that's definitely what the dendo and resulting graph seem to imply
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(proj)

    #Make and show a new scatter plot with the k-means clustering
    kmeans_3d = plt.figure()
    k_ax = kmeans_3d.add_subplot(111, projection='3d')
    label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
    k_ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=label_color, marker='o')
    plt.show()

    #Print size of clusters from lowest to highest (answer to question 1)
    print(sorted([kmeans.labels_.tolist().count(x) for x in range(min(kmeans.labels_), max(kmeans.labels_) + 1)]))
    
    # Make a nice dataframe so we can look at eigenvalues and their associated foods
    # (All these are intended to be done in Pycharm's pretty debugger or other similar places)
    eigframe = pd.DataFrame(eig_vecs, columns=raw_data.columns)

    #Take sums of absolute values of each columns eigenvectors (first few eigenvecs) to see which are most important
    feature_importances = sorted([(column, sum([abs(item) for item in eigframe[column].tolist()])) for column in eigframe.columns.tolist()[5:]], key= lambda x: x[1], reverse=True)

    #Now time for k-NN
    k_NN_shoppers = pd.read_csv("HW09_CLASSIFY_THESE_2185.csv", header=None)
    k_NN_shoppers = k_NN_shoppers.drop(k_NN_shoppers.columns[0], axis=1)

    #This monstrosity gives us a list of tuples for each item in the k-NN csv that needs to be classified
    #Each item in the list is a tuple with (distance to closest k-NN point, index of aforementioned point)
    closest_points = [min([(distance.euclidean(k_NN_shoppers.iloc[i], raw_data.iloc[x]), x) for x in range(len(list(raw_data.iterrows())))], key=lambda item: item[0]) for i in range(len(list(k_NN_shoppers.iterrows())))]

    # Add labels to original dataframe
    raw_data["Classifications"] = kmeans.labels_

    # We then simply use the indexes of closest_points to get a list of classifications for our k-NN test data
    k_NN_classifications = [raw_data.iloc[point[1]][-1] for point in closest_points]
    
    # Give Labels to the classified data
    shopper_labels = ["Family", "Vegan", "Hispanic Food", "Party Animal", "Glutten Free", "Kosher"]
    
    #print("Final label")
    print([shopper_labels[data] for data in k_NN_classifications])

main()

