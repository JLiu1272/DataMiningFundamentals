#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:59:15 2019

@author: JenniferLiu
File: HW_Liu_Jennifer_kMeans.py
"""

# Used for plotting graphs
import matplotlib.pyplot as plt

# Used to read CSV files 
import pandas as pd

# Used for numpy stuff 
import numpy as np 



import time 

class KMeans(object):
    def __init__(self, file):
        # Create a DataFrame object out of 
        # the K-Means value
        self.data = pd.read_csv(file, header=None, index_col = False)
        
        # Drop the index column
        self.data = self.data.drop(self.data.index[0], axis=1)
        
        # When the centroid move less than this threshold, stop 
        # clustering 
        self.threshold = 0.0005
    
    def plot_graph(self, x, y, title, xtitle, ytitle, colors=False, centroids=[]):
        """
        Plots the points and divide these points into
        clusters based on the kmeans algoritm result
        Plot he centroids as well 
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        # If color is set to True
        # Plot a graph showing how the clusters were split 
        if colors:
            diff_clusters = []
            
            # Split the clusters into their respective x and y component 
            for elm in range(self.k):
                
                # all x values
                x_points = [xi[0] for xi in np.extract(y[:,2] == elm, y[:,1])]
                
                # all y values 
                y_points = [yi[1] for yi in np.extract(y[:,2] == elm, y[:,1])]
                diff_clusters.append([x_points, y_points])
            

            colors = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#800000", "#808000", 
                       "#008000", "#FA8072", "#8A2BE2", "#4B0082", "#FF1493", "#D2691E", "#8B4513", "#BC8F8F", "#778899"]
            
            # Switch the colors as it moves to another cluster 
            for color_idx, cluster in enumerate(diff_clusters):
                if len(cluster) > 0:
                    plt.scatter(cluster[0], cluster[1], label='k = {}'.format(self.k),color=colors[color_idx])

            
            # Plot Centroids
            plt.scatter([x[0][0] for x in centroids], [y[0][1] for y in centroids], marker='o', color="#778899")
            
        else:
            # Plot data points with color
            ax.scatter(x, y, alpha=0.8, c="green", edgecolors='none', s=50)
        
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.show()
    
    def reset_vars(self, k):
        """
        Set all clusters back to start
        Set prev centroids to nothing
        """
        
        # Keep track of which centroid this point is closest to 
        # The index represent the point. The val of index represent
        # which centroid it corresponds to 
        # Initially all points are assigned to centroid 1 
        self.k = k 
        self.clusters = np.array([[float("inf"), 0, 0]]*self.data.shape[0], dtype=object)
        self.prev_centroids = np.array([])
    
    def reset_centroids(self, k):
        """
        Random selects new sets of k centroids 
        """
        self.centroids_idx = np.random.choice(self.data.shape[0], k)
        #self.centroids_idx = [0, 3]
        self.centroids = [np.array([np.array(self.data.iloc[idx]), 0]) for i,idx in enumerate(self.centroids_idx)]
        


    def find_closest_pair(self):
        """
        Compute every point to every centroid
        Check to see if there is a cluster that is better for
        that point 
        
        Returns an updated cluster information 
        """
        k = self.k
        
        # Regenerate clusters
        
        for elm in range(k):
            
            # centroid of this cluster 
            centroid = self.centroids[elm][0] 
            
            # Traverse every point 
            for p in range(self.data.shape[0]):
                
                # access the p row elm
                p_row = np.array(self.data.iloc[p])
                
                # euclidean dist
                new_dist = np.linalg.norm(centroid-p_row)
                
                # old euclidean dist 
                curr_dist = self.clusters[p][0]
                
                if new_dist < curr_dist:
                    self.clusters[p] = [new_dist, p_row, elm]
                           
        return self.clusters
        
    def compute_new_centroid(self):
        """
        Function for computing new centroid after values have been 
        reassigned 
        """
        
        # Update previous centroids
        self.prev_centroids = self.centroids
        
        # Get how many clusters there are 
        k = self.k
        
        # Generate new sets of centroids
        self.centroids = [0]*k
 
        # Total 
        for elm in range(k):
            # Traverse through clusters
            #all_elm = np.array([np.array(xi[1]) for i, xi in enumerate(self.clusters) if self.clusters[i, 2] == elm])
            all_elm = np.extract(self.clusters[:,2] == elm, self.clusters[:,1])
            
            # Total number of elements in this cluster elm
            total = all_elm.shape[0]
            
            # If the cluster is empty, do not compute 0 
            if total == 0:
                # Grab the old cluster, and use that
                self.centroids[elm] = [np.array(self.prev_centroids[elm][0]), 0]
                continue
            
            # The new centroid point for this cluster
            new_point = np.mean(all_elm, axis=0)
            
            # Append this new centroid data to a temp holder
            self.centroids[elm] = [new_point, total]

            
        # Convert list into np.array
        self.prev_centroids = np.array(self.prev_centroids)
        self.centroids = np.array(self.centroids)
            

    def stop_clustering_func(self):
        
        # If the previous centroid is empty, return False
        # it means this is our first cluster
        if self.prev_centroids.shape[0] == 0:
            return False 
        
        centroids_diff = np.linalg.norm(np.array(self.centroids[:,0]) - np.array(self.prev_centroids[:,0]))
        
        #print(np.array((np.array(centroids_diff) < self.threshold)).all())
        if np.array((np.array(centroids_diff) < self.threshold)).all() == True:
            return True 
        
        return False 
    
    def start_clustering(self, n, k):
        """
        Run the clustering main function
        
        n = # of iterations to run, each time generating a random centroids 
        k = # of clusters to make 
        """        
        
        # Data that shows the best SSE found 
        # for each of the k value 
        sse_v_k = [0]*k
        
        # shows the time it took to run the algo
        time_v_k = [0]*k
        
        # cluster centroids 
        centroids_v_k = [0]*k 
        
        # clusters 
        clusters_v_k = [0]*k
        
        best_cluster = None 
        best_centroids = None
        
        for k_idx in range(1,k+1):
            
            # For purpose of seeing progress
            print("Running: " + str(k_idx))
        
            min_sse = float("inf")
            total_time = None 
            
            for count in range(n):
                # For purpose of seeing progress
                print("\tIter: " + str(count))
                
                # Reset all centroids and all clusters
                self.reset_vars(k_idx)
                
                # Generate random centroids 
                self.reset_centroids(k_idx)
                
                #print(self.centroids)
                
                # If the stopping criteria for clustering has not been 
                # met, keep clustering
                # time how long e inner loop took
                start_time = time.time()
                while self.stop_clustering_func() is False:
                    self.find_closest_pair()
                    self.compute_new_centroid()
                    #print(self.centroids)
                               
                new_sse = np.sum([self.compute_SSE(e, self.clusters, self.centroids[e]) for e in range(k_idx)])
                end_time = time.time()
                
                # Update clusters, centroids if a better sse is found
                if new_sse < min_sse:
                    best_cluster = self.clusters
                    best_centroids = self.centroids 
                    min_sse = new_sse
                    
                    total_time = end_time - start_time
            
            # add these values to the respective bins 
            sse_v_k[k_idx-1] = min_sse
            time_v_k[k_idx-1] = total_time
            centroids_v_k[k_idx-1] = best_centroids
            clusters_v_k[k_idx-1] = best_cluster
            
            
        """
        Print Centroid statistics 
        """
        for elm in range(k_idx):
            print("************ K = " + str(elm+1) + " ***************")
            curr_centroids = centroids_v_k[elm]
            curr_cluster = clusters_v_k[elm]
            
            #[centroid for centroid in centroids_v_k[elm] if centroid[1] == elm][0]
            for i,cluster in enumerate(curr_centroids):
                print("Cluster ID:", end=" ")
                print(i + 1)
                                
                print("Centroids:", end=" ")
                # Print the centroid corresponding to this element 
                sorted_centroids = np.sort(cluster[0])
                print("{:0.1f}, {:0.1f}, {:0.1f}, {:0.1f}".format( \
                        sorted_centroids[0], \
                        sorted_centroids[1], \
                        sorted_centroids[2], \
                        sorted_centroids[3]))
                
                print("Num Points:", end=" ")
                print(cluster[1])
                
                print("SSE for this cluster")
                print(self.compute_SSE(i, curr_cluster, curr_centroids[i]))
                print()
            
            print("Overall Best SSE: ")
            print(sse_v_k[elm])
            print()

        
        print("-------- Graphing ---------------")
        """
        Plot graph showing the relationship between SSE and k 
        """
        kth = [(elm+1) for elm in range(k)]
        
        # Plot the graph showing relationship between SSE vs. K 
        self.plot_graph(kth, sse_v_k, "SSE vs. K", "K", "SSE")
        
        # Plot the graph
        self.plot_graph(kth, time_v_k, "Time vs. K", "K", "Time (s)")
        self.plot_graph(kth, best_cluster, "Best Clusters", "x", "y", colors=True,centroids=best_centroids)
        
        return sse_v_k, time_v_k

    def compute_SSE(self, k, clusters, k_centroid):
        """
        Generate sum of squared errors for the kth cluster 
        Param:
            k - which cluster to generate SSE for
            clusters - the cluster information
            k_centroids - the kth centroid info
        """
        
        # Points that are part of the cluster 
        kth_points = np.extract(clusters[:,2] == k, clusters[:,1])
        
        # Compute the sse for kth cluster
        k_sse = np.sum([np.linalg.norm(np.array(k_centroid[0])- k_p)**2 for k_p in kth_points])
        
        # return sse
        return k_sse

def main():
    
    k = 15
    n = 1000
    kmeans = KMeans("HW_K_MEANS__DATA_v2185.csv")
    #kmeans = KMeans("test2_kmeans.csv")
    
    #kmeans.reset_vars(k)
    #kmeans.reset_centroids(k)
    #print(kmeans.centroids)
    #kmeans.find_closest_pair()
    #kmeans.compute_new_centroid()
    #print(kmeans.stop_clustering_func())
    #print(kmeans.compute_SSE(0))
    
    #while kmeans.stop_clustering_func() is False:
    #    kmeans.find_closest_pair()
    #    kmeans.compute_new_centroid()
    #    print(kmeans.centroids)
        
    
    # Param K = 15, N times = 1000
    # start_cluster(n, k)
    kmeans.start_clustering(n, k)
    
    
    

main()