#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:23:33 2019

@author: Jennifer Liu, Joeseph Golden
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Dendogram 
from scipy.cluster.hierarchy import dendrogram, linkage

class Agglomeration(object):
    def __init__(self, csv):
        
        # A mtx representing distance
        # between 2 points 
        self.dist_mtx = None
        
        # Load CSV 
        self.cities = pd.read_csv(csv)
        
        # total num of cities
        self.num_cities = self.cities.shape[0]
        
        # The final cluster information 
        self.clusters = []
        
        # The cities that were clustered together
        self.cities_name = []

    def haversine(self, coord1, coord2):
        """
        Compute the haversine distance between
        coord1 and coord2
        """
        
        # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
        lon1, lat1 = coord1[0], coord1[1]
        lon2, lat2 = coord2[0], coord2[1]
    
        R = 6371000  # radius of Earth in meters
        phi_1 = math.radians(lat1)
        phi_2 = math.radians(lat2)
    
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
    
        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        meters = R * c  # output distance in meters
        km = meters / 1000.0  # output distance in kilometers
        #km = round(km, 3)
        
        return km

    def dist_matrix(self):
        """
        Read the lat,lon,city,country informations.
        Generate a distance matrix - a matrix that
        represents the haversine distance between p1 and p2. 
        """
        
        # Load CSV 
        cities = self.cities
        
        # Total number of cities in data
        num_cities = cities.shape[0]
        
        # Distance matrix
        distances = [[False for city in range(num_cities+1)] for city in range(num_cities)]
        
        # Traverse through every coordinate, and    
        # calculate distance to every other point except for
        # itself
        for coord1_i in range(num_cities):
            
            # Info about coord1
            coord1 = cities.iloc[coord1_i]
            
            # the first cell in row is the coord 1 object
            # 1st param contains a list of all points within this cluster
            # 2nd param contains a tuple indicated the cluster center 
            distances[coord1_i][0] = [[coord1], (coord1["Lon"], coord1["Lat"])]  
            
            for coord2_i in range(coord1_i+1, num_cities):
                
                # Info about coord1
                coord1 = cities.iloc[coord1_i]
                
                # Info about coord2
                coord2 = cities.iloc[coord2_i]
                
                # Compute haversine distance, and store in matrix 
                distances[coord1_i][coord2_i+1] = self.haversine((coord1["Lon"],coord1["Lat"]), \
                                                               (coord2["Lon"], coord2["Lat"]))
        
        # Update self matrix 
        self.dist_mtx = distances 

    def single_linkage(self):
        """
        Find the minimum distance between two point
        """
        
        # Update the min values everytime
        # a smaller distance is found
        min_dist = float("inf")
        min_row, min_col = 0, 0
        
        
        distances = self.dist_mtx
        
        # Traverse through the distances and
        # locate the minimum distance
        # note its row, and col, and the min distance 
        for row in range(len(distances)-1):
            for col in range(row+1, len(distances)):
                
                # Found a distance that is more minimal 
                if distances[row][col] < min_dist and distances[row][col] != False:
                    
                    # Save these values 
                    min_dist = distances[row][col]
                    min_row, min_col = row, col 
        
        return min_row, min_col, min_dist
    
    def merge(self, min_row, min_col):
        """
        Merge the two points that have the smallest distance 
        """
        
        coord1 = self.dist_mtx[min_row][0]
        coord2 = self.dist_mtx[min_col-1][0]
        
        # Grab the info for this coord
        coord1_det = coord1[0]
        coord2_det = coord2[0]
        
        # The total points in graph 
        total_points = len(coord1_det) + len(coord2_det)
        
        # compute new center of mass
        new_cluster_lat = (sum([p[2] for p in coord1_det]) + sum([p[2] for p in coord2_det]))/total_points
        new_cluster_lon = (sum([p[3] for p in coord1_det]) + sum([p[3] for p in coord2_det]))/total_points
        new_mass = (new_cluster_lat, new_cluster_lon)
        
        # Update cluster to include points from the two clusters
        coord1_det.extend(coord2_det)
        new_clusters = [[coord1_det, new_mass]]
        
        # Remove rows with min indexes 
        clusters_rm = np.delete(np.array(self.dist_mtx), (min_row, min_col-1), axis=0)
        
        # Remove col with min indexes 
        clusters_rm = np.delete(np.array(clusters_rm), (min_row+1, min_col), axis=1)
        
        # Update the distance matrix to the one removed
        self.dist_mtx = clusters_rm
        
        print("Num Clusters Remain: ")
        print(len(self.dist_mtx))
        
        new_filler = [False]*len(self.dist_mtx)
        self.dist_mtx = np.insert(self.dist_mtx, 1, new_filler , axis=1)
        
        filler = [False]*(len(self.dist_mtx)+1)
        new_clusters.extend(filler)
        #filler.insert(0, new_clusters)
        
        self.dist_mtx = np.insert(self.dist_mtx, 0, new_clusters, axis=0)
        
        #print(self.dist_mtx)
        
        #self.dist_mtx.append(new_clusters)
        
        self.compute_newdist()
        
        #self.print_distances()


    def compute_newdist(self):
        """
        Recompute the distance between the center of mass for the 
        merged cluster to every other point

        Get ready to see some really confusing list comprehensions
        """
        distances = self.dist_mtx
        
        # The coordinates for the points in the newly merged cluster
        start_points = [(coord["Lon"], coord["Lat"]) for coord in self.dist_mtx[0][0][0]]

        # A list of lists, each being all the points in one of the other clusters
        end_clusters = [[(point["Lon"], point["Lat"]) for point in clust[0][0]] for clust in self.dist_mtx.tolist()]

        for idx in range(1, len(end_clusters)):
            # Compute the distance between newly merged cluster
            # and all other clusters, and update them
            end_cluster = end_clusters[idx]
            distances[0][idx+1] = min([self.haversine(start_point, end_point) for start_point in start_points for end_point in end_cluster])
   
        
    def start_cluster(self, n):
        """
        Continuously cluster
        Param:
            n - the number of clusters we want 
        """
        idx = 1
        
        print("****** Iter: " + str(0) + " **********")
        #self.print_distances()
        
        # Stop clustering when there are only 4 clusters
        # remaining 
        while len(self.dist_mtx) > n:
            print("****** Iter: " + str(idx) + " **********")
            min_row, min_col, min_dist = self.single_linkage()
            print("Removing: ({}, {}) -- Distance: {}".format(min_row, min_col, min_dist))
            
            # Locate point with the smallest distance, and merge
            # them together, and calculate new distances
            self.merge(min_row, min_col)
            
            #print("Num Clusters:")
            #print(len(self.dist_mtx))

            self.print_distances()


            self.format_cities()
            
            idx += 1
        
        # Only extract the long and lat values for each point in
        # the cluster
        #self.format_cities()
        
    def format_cities(self):
        """
        A function to get the long and lat 
        of each of the cities
        
        CLusters points in this format - (Long, Lat)
        City - (City, Country)
        """
        
        clusters = []
        cities = []
        
        
        # Traverse through the cities info
        for row in range(len(self.dist_mtx)):
            cluster = []
            city = []
            
            # The datastructure will inform you
            # whether it is clustered or not 
            for coord in self.dist_mtx[row][0][0]:
                cluster.append((coord[3], coord[2]))
                city.append((coord[0].strip(), coord[1].strip()))
            
            clusters.append(cluster)
            cities.append(city)
        
        # Debugging 
        for i, coord in enumerate(clusters):
            print("Info: {} \nCoordinates: {}\n".format(cities[i], coord))     
        
        # Update the clusters and cities
        self.clusters = clusters
        self.cities_name = cities 

    def print_distances(self):
        #The "with" just sets pd options so it displays the whole matrix instead of using ... and displaying a portion
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
            #Use pandas dataframe to print a formatted table of distances
            display_df = pd.DataFrame([row[1:] for row in self.dist_mtx])
            display_df.insert(0, 'Cluster', [" and ".join([ item['City'].strip() + ", " + item['Country'].strip() for item in list(row)[0][0]]) for row in self.dist_mtx] )
            print(display_df)
        '''
        distances = self.dist_mtx

        for row in range(len(distances)):
            #print(distances[row])

            for col in range(1, len(distances)+1):                
                print("{0:.2f}".format(distances[row][col]), end=" ")
                
            print()

        '''

    
    def gen_dendogram(self, file):
        """
        Function to generate the dendogram 
        """
        
        # Create a data set with only long and latitutde 
        # [(lon, lat)....] etc 
        raw_data = self.cities.drop(["City", "Country"], axis=1)
        #raw_data = raw_data.set_index("Lon")
        
        columns = ["Lon", "Lat"]
        raw_data = raw_data.reindex(columns=columns)

        # Create labels for Dendogram
        labellst = [(self.cities["City"][idx].strip(),self.cities["Country"][idx].strip()) for idx in range(len(raw_data))]
        
        # Adjust size of plot 
        graph = plt.figure(figsize=(15, 12))
        
        # Create the dendogram using ward linkage 
        Z = linkage(raw_data, method='single', metric=lambda u, v: self.haversine(u, v) )
        dendrogram(Z, leaf_rotation=0, orientation='right', leaf_font_size=8, labels=labellst)
        #dendrogram(Z, leaf_rotation=0, orientation='right', leaf_font_size=8, labels=raw_data.index)
        plt.show() 
        
        # Save dendogram to a file named [file]
        graph.savefig(file)

   
def create_world(title, file, clusters):
    """
    Graph the world, and save it to a file
    given by the parameter
    
    Param:
        file - name of file 
    """
    graph = plt.figure(figsize=(12,6))
    
    m=Basemap(llcrnrlon=-180, llcrnrlat=-90,urcrnrlon=180,urcrnrlat=90)
    m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')
    m.drawcoastlines(linewidth=0.1, color="white")
    
    colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'pink', 'orange', 'violet', 'purple', 'black', 'white', 'gray']
    
    for idx, cluster in enumerate(clusters):
        
        lons = [lon[0] for lon in cluster]
        lats = [lat[1] for lat in cluster]
        
        x, y = m(lons, lats)
        
        m.plot(lons, lats, linestyle='none', marker="o", markersize=6, alpha=0.6, color=colors[idx], markeredgecolor="black", markeredgewidth=1)
        #m.scatter(x, y, marker="o", s=50, color=colors[idx], edgecolors="black")
    
    #lons = [0, 10, -20, -20]
    #lats = [0, -10, 40, -20]
    
    #x, y = m(lons, lats)
    
    #m.scatter(x, y, marker='D',color='m')

    plt.title(title)
    plt.show()
    graph.savefig(file)

def plot_points(clusters):
    """
    Plot the points on the map 
    give the points from CSV file 
    """
    
    create_world("Agglomeration Clusters", "agglomeration.png")
    pass
         

def main():
    """
    Function that calls function in 
    the correct sequence
    """
    
    #csv = "CS_420_City_Country_Lat_Lon_Shrt.csv"
    csv = "CS_420_City_Country_Lat_Lon.csv"
    
    agg = Agglomeration(csv)
    
    agg.dist_matrix()
    agg.print_distances()
    #print(agg.dist_mtx[1][1])
    
    agg.start_cluster(13)
    
    create_world("A Map of World", "agglomeration.png", agg.clusters)
    
    agg.gen_dendogram("MapDendogram.png")
    
    
main()

