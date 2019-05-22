
# coding: utf-8

# In[16]:


# Library for computing linear algebra math
import numpy as np 

# Library to do simple arithmetic operations, and do
# rounding (ceil, floor)
import math

# Library for loading a csv file and converting
# it into a dictionary
import pandas as pd 

# Library for displaying graphs
import matplotlib.pyplot as plt 


# In[18]:


"""
Load data (DATA_v525_FOR_CLASSIFICATION_using_Threshold_REVISED) into a dataframe
"""
car_speed_data = pd.read_csv('DATA_v525_FOR_CLASSIFICATION_using_Threshold_REVISED.csv')

# Display 4 rows of the data to check that
# the data successfully loaded into a dictionary structure
car_speed_data.head()


# In[19]:


"""
Round speeds to the nearest 0.5 mph
"""
# Set bin size to 0.5
bin_size = 0.5

# Bin the data points 
car_speed_data['Speed'] = np.round(car_speed_data['Speed']/bin_size)*bin_size

# Sort the car speed from slowest to fastest 
sorted_carspeed = car_speed_data.sort_values(by='Speed')

# Test that the sorting succeeded 
sorted_carspeed.head()


# In[20]:


# Determine the range of speeds available, and
# this range of number become all possible 
# thresholds

# First threshold to visit 
min_speed = math.floor(np.min(sorted_carspeed['Speed']))

# Last threshold to visit
max_speed = math.ceil(np.max(sorted_carspeed['Speed'])) + 1

# All possible thresholds 
possible_thresholds = np.arange(min_speed, max_speed, 0.5)


# In[22]:


"""
Adopted methodology from professor's lecture

Method determines the best threshold setting that would 
minimize # of misclassification 
"""
def best_threshold_func(data):
    
    # Determine the range of speeds available, and
    # this range of number become all possible 
    # thresholds
    max_speed = math.ceil(np.max(data['Speed'])) + 1
    
    # Noting the best threshold. Initialised to min speed
    min_speed = best_threshold = math.floor(np.min(data['Speed']))
    
    # Initialise best missclass rate to infinity
    best_cost = float("inf") 
    
    last_FN = 0
    last_FP = 0
    
    # Used to identify the point on
    # ROC Curve that has the lowest cost
    # function
    low_FPR = 0
    low_TPR = 0
    
    # Variable for keeping track of all false positive rate
    # FPR = FP / FP + TN 
    FPR_lst = []
    
    # Variable for keeping track of all true positive rate 
    TPR_lst = []
    
    # Store all of the cost functions generated 
    cost_funcs = []
    
    # Loop through every possible threshold that
    # is in the range of this dataset 
    for threshold in possible_thresholds:
        # Initialize True Positive (TP), True negative (TN), False Negative (FN), True Negative (TN)
        TN = TP = FN = FP = 0
        
        # Loop through all data, and find their TN, TP, FN, FP 
        # with the threshold 
        for index, person in data.iterrows():
            # Note person's speed
            c_speed = person['Speed']
            c_aggressive = person['Aggressive']
            if c_speed < threshold:
                # If aggressive = 1, then person trying to speed
                if c_aggressive == 1:
                    # Person was speeding, but detector didn't think so
                    FN += 1
                else:
                    # Person was not speeding, and detector didn't think so
                    TN += 1
            else:
                # If aggressive == 1, person speeding
                if c_aggressive == 1:
                    # Person was speeding, and detector caught it
                    TP += 1
                else:
                    # Person was not speeding, but detector think he/she is
                    FP += 1
        
        # Append FPR value for this threshold
        FPR_lst.append(float(FP)/float(FP+TN))
        
        # Append TPR value for this threshold
        TPR_lst.append(float(TP)/float(TP+FN))
        
        # Cost function 
        cost_func = FN + FP;
        regularization = cost_func - (FN + FP);
        cost_funcs.append(cost_func)
        
        # If found a cost function smaller than current, 
        # replace it as smallest, and mark its threshold
        if ( cost_func <= best_cost):
            best_cost = cost_func
            best_threshold = threshold
            # Keep the best FN and FP 
            last_FN = FN
            last_FP = FP
            # Determine best point on ROC Curve 
            low_FPR = float(FP)/float(FP+TN)
            low_TPR = float(TP)/float(TP+FN)
            #print("Regularization: {}, Objective Function: {}".format(regularization, FN+FP))
        
    return best_cost, best_threshold, last_FN, last_FP, cost_funcs, TPR_lst, FPR_lst, low_TPR, low_FPR

# Print the results
best_cost, best_threshold, last_FN, last_FP, cost_funcs, TPR_lst, FPR_lst, low_TPR, low_FPR = best_threshold_func(sorted_carspeed)
print("Best Cost: {}, Best Threshold: {}, FN: {}, FP: {}".format(best_cost, best_threshold, last_FN, last_FP))


# In[23]:


# Using matplotlib.pyplot lib, plot the cost function 
# as a function of the threshold use

# Initiating figure size of plot
plt.figure(figsize=(15,8), dpi=120)

# Label the axes
plt.title("Cost Function as Function of Threshold")
plt.xlabel("Speed (mph) -> Threshold")
plt.ylabel("Cost Function")
# X: Speed (mph), Y: Cost functions
plt.scatter([threshold for threshold in possible_thresholds], cost_funcs, s=120)
plt.show()


# In[24]:


"""
Bonus Point:
Generate a receiver-operator (ROC) curve for this training data. Plot it, and put the 
location of the any thresholds on the ROC curve. Label the X and Y axes correctly.
"""

# Initiating figure size of plot
plt.figure(figsize=(15,8), dpi=120)

# Label the axes
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# Plot the ROC Curve
plt.plot(FPR_lst, TPR_lst, '-o')

# Plot a straight line to demonstrate
# worse ROC curve possible
plt.plot([0, 1], [0, 1],'r--')

# Plot the point where cost is lowest 
plt.plot(low_FPR, low_TPR, '-o', color='orange')

# Anchor Graph to 0,1, so it is as square as possible 
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()


# In[ ]:




