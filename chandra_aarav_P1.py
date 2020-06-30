#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:00:56 2020
@author: aarav
"""

###### Importing necessary modules ######
# import random
import numpy as np


###### Operations for datafile input, randomisation, division, loading ######

# g = 'BikeAll.txt'

# inp_file = open(g,'r')
# # Reading first line of the file
# x = inp_file.readline()
# # Removing '\n' and extracting information by splitting at '\t'
# x = x.strip('\n')
# x = x.split('\t')
# # Storing the total number of rows and columns of the whole data
# total_rows = int(x[0])
# total_columns = int(x[1])

# # List comprehension for randomising data
# data = [ (random.random(), line) for line in inp_file ]
# # Sorting data based on random values generated above
# data.sort()
# inp_file.close()

# # Writing total row and column values in each the 3 files
# t_f = open('chandra_aarav_Train.txt','w')
# t_f.write("439\t11\n")
# v_f = open('chandra_aarav_Valid.txt','w')
# v_f.write("146\t11\n")
# te_f = open('chandra_aarav_Test.txt','w')
# te_f.write("146\t11\n")

# ###### Dividing the data into Train, Validate and Test sets ######
# for i in range(int(total_rows)):
#     # Train Data
#     if (i<439):
#         t_f.write(data[i][1])
#     # Validation Data
#     elif(i>=439 and i<585):
#         v_f.write(data[i][1])
#     # Test Data
#     else:
#         te_f.write(data[i][1])
    
# t_f.close()
# v_f.close()
# te_f.close()

# Normal Equation Method:
def NormalEqnMethod(x,y): 
    return np.dot(np.linalg.pinv(np.dot(x.T, x)),np.dot(x.T, y))
    
# For getting h(x):
def getHypothesisNEM(x,w1):
    return np.dot(x,w1)

# For getting error:
def getErrorNEM(hw,y):
    return (np.power(np.subtract(hw,y),2))

# For getting J:
def getMSE_NEM(error,m1,m):
    q = np.dot(m1.T,error)
    return ((np.sum(q))/(2*m))
    #return(np.sum(error)/2*m)

# For getting R^2:
def getRsquare(m_s_e,y):
    return(1-(m_s_e/(np.sum(np.power(y-(np.full(np.shape(y),np.mean(y))),2))/(2*np.shape(y)[0]))))

# For getting adjusted R^2:
def getAdjustedRsquare(square_error,m,y,n):
    r_square = getRsquare(square_error,y)
    return (1 - ((1-r_square)*(m-1))/(m-n-1))

########## Getting the Training data ##########
g = input("Enter the Train data filename without .txt: ") 
g = g.strip()+'.txt'
# g = 'chandra_aarav_Train.txt'
inp_file = open(g,mode ='r')
x = inp_file.readline()
x = x.strip('\n')
x = x.split('\t')
# Storing the total number of rows and columns of the Train data file.
t_rows = int(x[0])
t_columns = int(x[1])
# Creating a list of list for all the Train data (only values)
t_tot=[(inp_file.readline().strip('\n').split('\t')) for i in range(int(t_rows))]
inp_file.close()

t_tot = np.array(t_tot, dtype=float)
# Storing the shape to extract feature columns and result column
t_shape = np.shape(t_tot)
# Extracting all columns except the last column as 'features'
t_X = t_tot[:,:t_shape[1]-1]
# Extracting only the last column as 'result'
t_Y = t_tot[:,t_shape[1]-1]
t_Y = np.reshape(t_Y, (t_rows,1))

###### Creating Weight array with initial weights ######
gc = t_tot.shape[1]
gr = t_tot.shape[0]
m1 = np.ones(np.shape(t_Y))

# Appending x0 = 1 value as instructed in the slides in the test dataset.
x0 = np.ones((gr,1))
# adding ones at the beginning.
train_X = np.hstack([x0,t_X])
# Calling the functions written above
w = NormalEqnMethod(train_X,t_Y)
w = np.reshape(w,(np.shape(train_X)[1],1))
hypo_t = getHypothesisNEM(train_X,w)
error_m_t = getErrorNEM(hypo_t,t_Y)
mean_squ_err_t = getMSE_NEM(error_m_t,m1,np.shape(t_Y)[0])
R_square = getRsquare(mean_squ_err_t,t_Y)
print ("\nWeights for the training model 'first':\n",w)
print ("\nJ for training data with the 'first' model weights = "+str(mean_squ_err_t))

# Creating 'Second' Model by adding the square values to the existing model:
train_X2 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),2),(gr,1))
    train_X2 = np.hstack([train_X2,x_add])
# adding ones at the beginning.
train_X2 = np.hstack([x0,train_X2])
# Calling the functions written above
weights2 = NormalEqnMethod(train_X2,t_Y)
weights2 = np.reshape(weights2,(np.shape(train_X2)[1],1))
hypo2_t = getHypothesisNEM(train_X2,weights2)
error_m2_t = getErrorNEM(hypo2_t,t_Y)
mean_squ_err2_t = getMSE_NEM(error_m2_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'Second':\n",weights2)
print ("\nJ for training data with the 'Second' weights = "+str(mean_squ_err2_t))

# Creating 'third' Model by adding the square values to the existing model:
train_X3 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),3),(gr,1))
    train_X3 = np.hstack([train_X3,x_add])
# adding ones at the beginning.
train_X3 = np.hstack([x0,train_X3])
# Calling the functions written above
weights3 = NormalEqnMethod(train_X3,t_Y)
weights3 = np.reshape(weights3,(np.shape(train_X3)[1],1))
hypo3_t = getHypothesisNEM(train_X3,weights3)
error_m3_t = getErrorNEM(hypo3_t,t_Y)
mean_squ_err3_t = getMSE_NEM(error_m3_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'third':\n",weights3)
print ("\nJ for training data with the 'third' weights = "+str(mean_squ_err3_t))


# Creating 'fourth' Model by adding the square values to the existing model:
train_X4 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),4),(gr,1))
    train_X4 = np.hstack([train_X4,x_add])


# adding ones at the beginning.
train_X4 = np.hstack([x0,train_X4])
# Calling the functions written above
weights4 = NormalEqnMethod(train_X4,t_Y)
weights4 = np.reshape(weights4,(np.shape(train_X4)[1],1))
hypo4_t = getHypothesisNEM(train_X4,weights4)
error_m4_t = getErrorNEM(hypo4_t,t_Y)
mean_squ_err4_t = getMSE_NEM(error_m4_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'fourth':\n",weights4)
print ("\nJ for training data with the 'fourth' weights = "+str(mean_squ_err4_t))

# Creating 'fifth' Model by adding the square values to the existing model:
train_X5 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.sqrt((t_X[:,i])),(gr,1))
    x_add = np.negative(x_add)
    train_X5 = np.hstack([train_X5,x_add])
# adding ones at the beginning.
train_X5 = np.hstack([x0,train_X5])
# Calling the functions written above
weights5 = NormalEqnMethod(train_X5,t_Y)
weights5 = np.reshape(weights5,(np.shape(train_X5)[1],1))
hypo5_t = getHypothesisNEM(train_X5,weights5)
error_m5_t = getErrorNEM(hypo5_t,t_Y)
mean_squ_err5_t = getMSE_NEM(error_m5_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'fifth':\n",weights5)
print ("\nJ for training data with the 'fifth' weights = "+str(mean_squ_err5_t))

# Creating 'sixth' Model by adding the square values to the existing model:
train_X6 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    train_X6 = np.hstack([train_X6,x_add])
    x_add = np.reshape(np.power((t_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    train_X6 = np.hstack([train_X6,x_add])
# adding ones at the beginning.
train_X6 = np.hstack([x0,train_X6])
# Calling the functions written above
weights6 = NormalEqnMethod(train_X6,t_Y)
weights6 = np.reshape(weights6,(np.shape(train_X6)[1],1))
hypo6_t = getHypothesisNEM(train_X6,weights6)
error_m6_t = getErrorNEM(hypo6_t,t_Y)
mean_squ_err6_t = getMSE_NEM(error_m6_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'sixth':\n",weights6)
print ("\nJ for training data with the 'sixth' weights = "+str(mean_squ_err6_t))

# Creating 'seventh' Model by adding the square values to the existing model:
train_X7 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    train_X7 = np.hstack([train_X7,x_add])
    x_add = np.reshape(np.power((t_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    train_X7 = np.hstack([train_X7,x_add])
    x_add = np.reshape(np.power((t_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    train_X7 = np.hstack([train_X7,x_add])
# adding ones at the beginning.
train_X7 = np.hstack([x0,train_X7])
# Calling the functions written above
weights7 = NormalEqnMethod(train_X7,t_Y)
weights7 = np.reshape(weights7,(np.shape(train_X7)[1],1))
hypo7_t = getHypothesisNEM(train_X7,weights7)
error_m7_t = getErrorNEM(hypo7_t,t_Y)
mean_squ_err7_t = getMSE_NEM(error_m7_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'seventh':\n",weights7)
print ("\nJ for training data with the 'seventh' weights = "+str(mean_squ_err7_t))




# Creating 'eight' Model by adding the square values to the existing model:
train_X10 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),3),(gr,1))
    train_X10 = np.hstack([train_X10,x_add])
    x_add = np.reshape(np.power((t_X[:,i]),4),(gr,1))
    train_X10 = np.hstack([train_X10,x_add])
# adding ones at the beginning.
train_X10 = np.hstack([x0,train_X10])
# Calling the functions written above
weights10 = NormalEqnMethod(train_X10,t_Y)
weights10 = np.reshape(weights10,(np.shape(train_X10)[1],1))
hypo10_t = getHypothesisNEM(train_X10,weights10)
error_m10_t = getErrorNEM(hypo10_t,t_Y)
mean_squ_err10_t = getMSE_NEM(error_m10_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model 'eight':\n",weights10)
print ("\nJ for training data with the 'eight' weights = "+str(mean_squ_err10_t))

# Creating ‘ninth’ Model by adding the square values to the existing model:
train_X12 = np.copy(t_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((t_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    train_X12 = np.hstack([train_X12,x_add])
    x_add = np.reshape(np.power((t_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    train_X12 = np.hstack([train_X12,x_add])
# adding ones at the beginning.
train_X12 = np.hstack([x0,train_X12])
# Calling the functions written above
weights12 = NormalEqnMethod(train_X12,t_Y)
weights12 = np.reshape(weights12,(np.shape(train_X12)[1],1))
hypo12_t = getHypothesisNEM(train_X12,weights12)
error_m12_t = getErrorNEM(hypo12_t,t_Y)
mean_squ_err12_t = getMSE_NEM(error_m12_t,m1,np.shape(t_Y)[0])
print ("\nWeights for the training model ‘ninth’:\n",weights12)
print ("\nJ for training data with the ‘ninth’ weights = "+str(mean_squ_err12_t))


########## Getting the Validation data ##########

g = input("Enter the Validation data filename without .txt: ") 
g = g .strip()+'.txt'
# g = 'chandra_aarav_Valid.txt'
inp_file = open(g,mode ='r')
x = inp_file.readline()
# Removing '\n' and extracting first line data of the Validation data file by splitting at '\t'. 
x = x.strip('\n')
x = x.split('\t')
# Storing the total number of rows and columns of the Validate data file.
v_rows = x[0]
v_columns = x[1]
# Creating a list of list for all the Validation data (only values)
v_tot=[(inp_file.readline().strip('\n').split('\t')) for i in range(int(v_rows))]
inp_file.close()

###### Converting to nparray and extracting the columns ######
v_tot = np.array(v_tot, dtype=float)
# Storing the shape to extract feature columns and result column
v_shape = np.shape(t_tot)
# Extracting all columns except the last column as 'features'
v_X = v_tot[:,:v_shape[1]-1]
# Extracting only the last column as 'result'
v_Y = v_tot[:,v_shape[1]-1]
v_Y = np.reshape(v_Y,(np.shape(v_Y)[0],1))

m1 = np.ones(np.shape(v_Y))
gc = v_tot.shape[1]
gr = v_tot.shape[0]

x0 = np.ones((gr,1))
# adding ones at the beginning.
valid_X = np.hstack([x0,v_X])
# Calling the functions written above
hypo_v = getHypothesisNEM(valid_X,w)
error_m_v = getErrorNEM(hypo_v,v_Y)
mean_squ_err_v = getMSE_NEM(error_m_v,m1,np.shape(v_Y)[0])
R_square_v = getRsquare(mean_squ_err_v,v_Y)
# print ("\nWeights for the linear validation model :\n",w)
print ("\nJ for validation data with the 'first' weights = "+str(mean_squ_err_v))

# Creating 'Quadratic' Model by adding the square values to the existing model:
valid_X2 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),2),(gr,1))
    valid_X2 = np.hstack([valid_X2,x_add])
# adding ones at the beginning.
valid_X2 = np.hstack([x0,valid_X2])
# Calling the functions written above
hypo2_v = getHypothesisNEM(valid_X2,weights2)
error_m2_v = getErrorNEM(hypo2_v,v_Y)
mean_squ_err2_v = getMSE_NEM(error_m2_v,m1,np.shape(v_Y)[0])  
print ("\nJ for validation data with the 'second' weights = "+str(mean_squ_err2_v))

# Creating 'third' Model by adding the square values to the existing model:
valid_X3 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),3),(gr,1))
    valid_X3 = np.hstack([valid_X3,x_add])
# adding ones at the beginning.
valid_X3 = np.hstack([x0,valid_X3])
# Calling the functions written above
hypo3_v = getHypothesisNEM(valid_X3,weights3)
error_m3_v = getErrorNEM(hypo3_v,v_Y)
mean_squ_err3_v = getMSE_NEM(error_m3_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'third' weights = "+str(mean_squ_err3_v))

# Creating 'fourth' Model by adding the square values to the existing model:
valid_X4 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),4),(gr,1))
    valid_X4 = np.hstack([valid_X4,x_add])
# adding ones at the beginning.
valid_X4 = np.hstack([x0,valid_X4])
# Calling the functions written above
hypo4_v = getHypothesisNEM(valid_X4,weights4)
error_m4_v = getErrorNEM(hypo4_v,v_Y)
mean_squ_err4_v = getMSE_NEM(error_m4_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'fourth' weights = "+str(mean_squ_err4_v))

# Creating 'fifth' Model by adding the square values to the existing model:
valid_X5 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.sqrt((v_X[:,i])),(gr,1))
    x_add = np.negative(x_add)
    valid_X5 = np.hstack([valid_X5,x_add])
# adding ones at the beginning.
valid_X5 = np.hstack([x0,valid_X5])
# Calling the functions written above
hypo5_v = getHypothesisNEM(valid_X5,weights5)
error_m5_v = getErrorNEM(hypo5_v,v_Y)
mean_squ_err5_v = getMSE_NEM(error_m5_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'fifth' weights = "+str(mean_squ_err5_v))

# Creating 'sixth' Model by adding the square values to the existing model:
valid_X6 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    valid_X6 = np.hstack([valid_X6,x_add])
    x_add = np.reshape(np.power((v_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    valid_X6 = np.hstack([valid_X6,x_add])
# adding ones at the beginning.
valid_X6 = np.hstack([x0,valid_X6])
# Calling the functions written above
hypo6_v = getHypothesisNEM(valid_X6,weights6)
error_m6_v = getErrorNEM(hypo6_v,v_Y)
mean_squ_err6_v = getMSE_NEM(error_m6_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'sixth' weights = "+str(mean_squ_err6_v))

# Creating 'seventh' Model by adding the square values to the existing model:
valid_X7 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    valid_X7 = np.hstack([valid_X7,x_add])
    x_add = np.reshape(np.power((v_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    valid_X7 = np.hstack([valid_X7,x_add])
    x_add = np.reshape(np.power((v_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    valid_X7 = np.hstack([valid_X7,x_add])
# adding ones at the beginning.
valid_X7 = np.hstack([x0,valid_X7])
# Calling the functions written above
hypo7_v = getHypothesisNEM(valid_X7,weights7)
error_m7_v = getErrorNEM(hypo7_v,v_Y)
mean_squ_err7_v = getMSE_NEM(error_m7_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'seventh' weights = "+str(mean_squ_err7_v))


# Creating 'eight' Model by adding the square values to the existing model:
valid_X10 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),3),(gr,1))
    valid_X10 = np.hstack([valid_X10,x_add])
    x_add = np.reshape(np.power((v_X[:,i]),4),(gr,1))
    valid_X10 = np.hstack([valid_X10,x_add])
# adding ones at the beginning.
valid_X10 = np.hstack([x0,valid_X10])
# Calling the functions written above
hypo10_v = getHypothesisNEM(valid_X10,weights10)
error_m10_v = getErrorNEM(hypo10_v,v_Y)
mean_squ_err10_v = getMSE_NEM(error_m10_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the 'eight' weights = "+str(mean_squ_err10_v))

# Creating ‘ninth’ Model by adding the square values to the existing model:
valid_X12 = np.copy(v_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((v_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    valid_X12 = np.hstack([valid_X12,x_add])
    x_add = np.reshape(np.power((v_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    valid_X12 = np.hstack([valid_X12,x_add])
# adding ones at the beginning.
valid_X12 = np.hstack([x0,valid_X12])
# Calling the functions written above
hypo12_v = getHypothesisNEM(valid_X12,weights12)
error_m12_v = getErrorNEM(hypo12_v,v_Y)
mean_squ_err12_v = getMSE_NEM(error_m12_v,m1,np.shape(v_Y)[0])
print ("\nJ for validation data with the ‘ninth’ weights = "+str(mean_squ_err12_v))






########## Getting the Test data ##########
g = input("Enter the Test data filename without .txt: ") 
g = g .strip()+'.txt'
# g = 'chandra_aarav_Test.txt'
inp_file = open(g,mode ='r')
x = inp_file.readline()
# Removing '\n' and extracting first line data of the Test data file by splitting at '\t'. 
x = x.strip('\n')
x = x.split('\t')
# Storing the total number of rows and columns of the Test data file.
te_rows = x[0]
te_columns = x[1]
# Creating a list of list for all the Test data (only values)
te_tot=[(inp_file.readline().strip('\n').split('\t')) for i in range(int(te_rows))]
inp_file.close()

###### Converting to nparray and extracting the columns ######
te_tot = np.array(te_tot, dtype=float)
# Storing the shape to extract feature columns and result column
te_shape = np.shape(t_tot)
# Extracting all columns except the last column as 'features'
test_X = te_tot[:,:te_shape[1]-1]
# Extracting only the last column as 'result'
test_Y = te_tot[:,te_shape[1]-1]
test_Y = np.reshape(test_Y,(np.shape(test_Y)[0],1))

m1 = np.ones(np.shape(test_Y))
gc = te_tot.shape[1]
gr = te_tot.shape[0]

x0 = np.ones((gr,1))
test_X1 = np.hstack([x0,test_X])
# Calling the functions written above
hypo_te = getHypothesisNEM(test_X1,w)
error_m_te = getErrorNEM(hypo_te,test_Y)
mean_squ_err_te = getMSE_NEM(error_m_te,m1,np.shape(test_Y)[0]) 
print ("\nJ for test data set (linear)= "+str(mean_squ_err_te)) 
print ("Adjusted R^2 for test data set (linear)= "+str(getAdjustedRsquare(mean_squ_err_te,te_shape[0],test_Y,te_shape[1])))

# Coverting the test data for use with the quadratic model
test_X2 = np.copy(test_X)
for i in range(gc-1):
    # it stores the square of every feature value
    x_add_2 = np.reshape(np.power((test_X[:,i]),2),(gr,1))
    # adding the squared values 
    test_X2 = np.hstack([test_X2,x_add_2])
    
# adding ones at the beginning.   
test_X2 = np.hstack([x0,test_X2])
# Calling the functions written above
hypo2_te = getHypothesisNEM(test_X2,weights2)
error_m2_te = getErrorNEM(hypo2_te,test_Y)
mean_squ_err2_te = getMSE_NEM(error_m2_te,m1,np.shape(test_Y)[0]) 
print ("\nJ for test data set (second)= "+str(mean_squ_err2_te)) 
print ("Adjusted R^2 for test data set (second)= "+str(getAdjustedRsquare(mean_squ_err2_te,te_shape[0],test_Y,te_shape[1])))

# Creating 'third' Model by adding the square values to the existing model:
test_X3 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),3),(gr,1))
    test_X3 = np.hstack([test_X3,x_add])
# adding ones at the beginning.
test_X3 = np.hstack([x0,test_X3])
# Calling the functions written above
hypo3_te = getHypothesisNEM(test_X3,weights3)
error_m3_te = getErrorNEM(hypo3_te,test_Y)
mean_squ_err3_te = getMSE_NEM(error_m3_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set (third)= "+str(mean_squ_err3_te)) 
print ("Adjusted R^2 for test data set (third)= "+str(getAdjustedRsquare(mean_squ_err3_te,te_shape[0],test_Y,te_shape[1])))

# Creating 'fourth' Model by adding the square values to the existing model:
test_X4 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),4),(gr,1))
    test_X4 = np.hstack([test_X4,x_add])
# adding ones at the beginning.
test_X4 = np.hstack([x0,test_X4])
# Calling the functions written above
hypo4_te = getHypothesisNEM(test_X4,weights4)
error_m4_te = getErrorNEM(hypo4_te,test_Y)
mean_squ_err4_te = getMSE_NEM(error_m4_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set (fourth)= "+str(mean_squ_err4_te)) 
print ("Adjusted R^2 for test data set (fourth)= "+str(getAdjustedRsquare(mean_squ_err4_te,te_shape[0],test_Y,te_shape[1])))


# Creating 'fifth' Model by adding the square values to the existing model:
test_X5 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.sqrt((test_X[:,i])),(gr,1))
    x_add = np.negative(x_add)
    test_X5 = np.hstack([test_X5,x_add])
# adding ones at the beginning.
test_X5 = np.hstack([x0,test_X5])
# Calling the functions written above
hypo5_te = getHypothesisNEM(test_X5,weights5)
error_m5_te = getErrorNEM(hypo5_te,test_Y)
mean_squ_err5_te = getMSE_NEM(error_m5_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set (fifth)= "+str(mean_squ_err5_te)) 
print ("Adjusted R^2 for test data set (fifth)= "+str(getAdjustedRsquare(mean_squ_err5_te,te_shape[0],test_Y,te_shape[1])))

# Creating 'sixth' Model by adding the square values to the existing model:
test_X6 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    test_X6 = np.hstack([test_X6,x_add])
    x_add = np.reshape(np.power((test_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    test_X6 = np.hstack([test_X6,x_add])
# adding ones at the beginning.
test_X6 = np.hstack([x0,test_X6])
# Calling the functions written above
hypo6_te = getHypothesisNEM(test_X6,weights6)
error_m6_te = getErrorNEM(hypo6_te,test_Y)
mean_squ_err6_te = getMSE_NEM(error_m6_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set (sixth)= "+str(mean_squ_err6_te)) 
print ("Adjusted R^2 for test data set (sixth)= "+str(getAdjustedRsquare(mean_squ_err6_te,te_shape[0],test_Y,te_shape[1])))


# Creating 'seventh' Model by adding the square values to the existing model:
test_X7 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    test_X7 = np.hstack([test_X7,x_add])
    x_add = np.reshape(np.power((test_X[:,i]),3),(gr,1))
    x_add = np.negative(x_add)
    test_X7 = np.hstack([test_X7,x_add])
    x_add = np.reshape(np.power((test_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    test_X7 = np.hstack([test_X7,x_add])
# adding ones at the beginning.
test_X7 = np.hstack([x0,test_X7])
# Calling the functions written above
hypo7_te = getHypothesisNEM(test_X7,weights7)
error_m7_te = getErrorNEM(hypo7_te,test_Y)
mean_squ_err7_te = getMSE_NEM(error_m7_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set 'seventh'= "+str(mean_squ_err7_te)) 
print ("Adjusted R^2 for test data set 'seventh'= "+str(getAdjustedRsquare(mean_squ_err7_te,te_shape[0],test_Y,te_shape[1])))




# Creating 'eight' Model by adding the square values to the existing model:
test_X10 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),3),(gr,1))
    test_X10 = np.hstack([test_X10,x_add])
    x_add = np.reshape(np.power((test_X[:,i]),4),(gr,1))
    test_X10 = np.hstack([test_X10,x_add])
# adding ones at the beginning.
test_X10 = np.hstack([x0,test_X10])
# Calling the functions written above
hypo10_te = getHypothesisNEM(test_X10,weights10)
error_m10_te = getErrorNEM(hypo10_te,test_Y)
mean_squ_err10_te = getMSE_NEM(error_m10_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set 'eight'= "+str(mean_squ_err10_te)) 
print ("Adjusted R^2 for test data set 'eight'= "+str(getAdjustedRsquare(mean_squ_err10_te,te_shape[0],test_Y,te_shape[1])))

# Creating ‘ninth’ Model by adding the square values to the existing model:
test_X12 = np.copy(test_X)
for i in range(gc-1):
    x_add = np.reshape(np.power((test_X[:,i]),2),(gr,1))
    x_add = np.negative(x_add)
    test_X12 = np.hstack([test_X12,x_add])
    x_add = np.reshape(np.power((test_X[:,i]),4),(gr,1))
    x_add = np.negative(x_add)
    test_X12 = np.hstack([test_X12,x_add])
# adding ones at the beginning.
test_X12 = np.hstack([x0,test_X12])
# Calling the functions written above
hypo12_te = getHypothesisNEM(test_X12,weights12)
error_m12_te = getErrorNEM(hypo12_te,test_Y)
mean_squ_err12_te = getMSE_NEM(error_m12_te,m1,np.shape(test_Y)[0])
print ("\nJ for test data set ‘ninth’= "+str(mean_squ_err12_te)) 
print ("Adjusted R^2 for test data set ‘ninth’= "+str(getAdjustedRsquare(mean_squ_err12_te,te_shape[0],test_Y,te_shape[1])))

