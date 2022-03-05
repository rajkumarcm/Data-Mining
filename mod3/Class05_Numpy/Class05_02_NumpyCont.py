# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# Numpy continued
import numpy as np 

#%%
# Stacking numpy arrays
a = np.floor(10*np.random.random((2,3)));print('a:',a)
b = np.floor(10*np.random.random((2,3)));print('b:',b)
# a=[1,2,3]
# b=[4,5,6]
z = np.vstack((a,b)); print('vstack z:',z)
z1 = np.hstack((a,b)); print('hstack z1:',z1)
z2 = np.column_stack((a,b)) 
print('column_stack z2:\n',z2)
# column_stack same as hstack between two numpy arrays here
# the two has different behavior acting on lists
# try comment in the python list defs for a and b above. 
print('#',50*"-")

#%%
# newaxis
from numpy import newaxis
a = np.array([4.,2.,6.])
b = np.array([3.,7.,5.])
print('a[:,newaxis] :\n',a[:,newaxis])
z3= np.column_stack((a[:,newaxis],b[:,newaxis]))
print('z3:\n',z3)
z3b= np.hstack((a[:,newaxis],b[:,newaxis]))
print('z3b:\n',z3b)
z4= np.vstack((a[:,newaxis],b[:,newaxis])); 
print('z4:\n',z4)
print('#',50*"-")

#%%
# Splitting Numpy arrays
a = np.floor(10*np.random.random((2,12)))
print('a:',a)
z1 = np.hsplit(a,3) # Split a into 3. If dimension not divisible -> ValueError
print('type(z1):',type(z1))
print('len(z1):',len(z1))
for elt in z1:
  print('elt:\n',elt)

#%%
# Split a after the third and the fourth column
print('a:',a)
z2 = np.hsplit(a,(3,4))
# print('type(z1):',type(z1))
print('len(z2):',len(z2))
for elt in z2:
  print('elt:\n',elt)
print('#',50*"-")

#%%
# The .ix_() function ( I interpret it as Index-eXtraction )
# https://numpy.org/doc/stable/reference/generated/numpy.ix_.html 
#
e = np.arange(25).reshape(5,5)
print('e:\n',e)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]]
print()

sub_indices = np.ix_([1,3],[0,4]) # If your ultimate ndarray is rank 3 (m x n x k), then you will need three arguments
print('type(sub_indices) :', type(sub_indices))
print('len(sub_indices) :', len(sub_indices)) # In other words, for rank 3 ndarray, you'll need tuple here of length 3.
print('sub_indices :\n', sub_indices)
print()

print('type(sub_indices[0]) :', type(sub_indices[0]))
print('sub_indices[0].shape :', sub_indices[0].shape)
print('type(sub_indices[1]) :', type(sub_indices[1]))
print('sub_indices[1].shape :', sub_indices[1].shape)

print('\ne[sub_indices]:\n', e[sub_indices])

#%%
# rank 3 case
a = np.array([4,2,3])
b = np.array([5,4])
c = np.array([5,4,0,3])
ax,bx,cx = np.ix_(a,b,c) # This separates out the [0], [1], [2] elements of the tuple in one step
print('ax:',ax); print('bx:',bx); print('cx:',cx)
print('shapes:', ax.shape, bx.shape, cx.shape)
result = ax+bx*cx
print('result:',result)
print('result[3,2,4]:',result[2,1,3])
print('individual:',a[2]+b[1]*c[3])

# More commonly, we use it for filtering like our previous rank 2 example.
e = np.arange(6*6*6).reshape(6,6,6)  #216 elements
print('\ne[(ax,bx,cx)]:\n', e[(ax,bx,cx)])



#%%
# Automatic Reshaping
a = np.arange(30)
print('Original- a.shape:',a.shape); print('Original- a:',a)
a.shape = 2,-1,3
print('After- a.shape:',a.shape); print('After- a:',a)
# Unlike the reshape() method, this way changes a directly!!
print('#',50*"-")
#%%
# Other functions worth learning:
# https://numpy.org/doc/stable/reference/generated/numpy.indices.html
idlabels =  np.indices( (3,4) ) 
# This creates a set of i, j values 
# indicating the row/column indices, with dimensions 3 by 4
i, j = idlabels
print('i:',i)
print('j:',j)
# We can try, for example, create a matrix of M, whose values are 3*i - 2*j
m = 2*i - 2*j
print('m:',m)

#%%
#

#%%
# Examples from https://www.machinelearningplus.com/python/101-numpy-exercises-python/ 
# and https://www.w3resource.com/python-exercises/numpy/index-array.php

#%%



