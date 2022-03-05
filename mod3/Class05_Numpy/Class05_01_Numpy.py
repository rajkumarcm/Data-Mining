# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

# # SciPy Family
#
# Python-based ecosystem [scipy.org](https://scipy.org)  
# 
# * SciPy Library - Fundamental library for scientific computing
# * NumPy - Numeric Python: Base N-dimensional array package
# * Pandas - Data structures (Dataframes) & analysis
# * Mathplotlib - Comprehensive 2D Plotting
# * Sympy - Symbolic mathematics
# * IPython - Enhanced Interactive Console
#
# The datatypes (dtype attribute) supported by Numpy is many:  
# [Numpy basic data types](https://docs.scipy.org/doc/numpy/user/basics.types.html) 

#%%
# might need to install numpy from the terminal
# %pip install numpy
# %pip3 install numpy
# %sudo pip install numpy
# %sudo pip3 install numpy
# %sudo -H pip3 install numpy
# %conda install numpy
# %pip freeze
# %pip list
# %pip show numpy

#%%
import numpy as np 
# or from numpy import *  
# import matplotlib.pyplot as plt
# import pandas as pd  

#%%
#
# Review lists
list0 = [9,8,7]
list0b = [6,5,4]
#
# What are the lengths of list1 and list1b?

#%%
# What do you get with list0 + list0b?
#
list0+list0b


#%%
# explore data structures with list of list, how many dimensions? 
list1 = [ [11,12,13,14], [21,22,23,24], [31,32,33,34]] 
list1b = [ [41,42,43,44], [51,52,53,54], [61,62,63,64]] 
#

#%%
# Again, what is list1 + list1b?
#
list1+list1b

#%%
# Question: How do you describe (in english) these two lists? What are the "shapes" of the objects?
#
# These are 3 by 4 matrices. Two-dimensional arrays. 
#
# Question: how do you get the element '32' in list1?
#
# 
# 
# Question: how do you get the row of [31,32,33,34] in list1?
# 
#
# 
#%%
# Question: How to you get the column of 12, 22, 32 ???
# 
#
#%%
[ row[1] for row in list1 ]

#%%
# OR Loop it
v3 = []
for row in list1: 
  v3.append(row[1])
print(v3)


#%%

#%%
list2 = [ [11,12,13], [21,22,23], [31,32,33], [41,42,43] ] # two dimensional list (2-D array)  # (4,3)
# list2b = [ [51,52,53], [61,62,63], [71,72,73], [81,82,83]] 


# How do you access the different parts of these two lists?

#%%
# How do you create a higher-dimensional list (say 2x3x4)?
# 
# list3D = [ [ [111,112,113], [121,122,123], [131,132,133], [141,142,143] ] 
# , [ [211,212,213], [221,222,223], [231,232,233], [241,242,243] ] ] 

list3D = [ [ [ 111, 112, 113, 114 ], [ 121, 122, 123, 124 ], [131, 132, 133, 134] ] , 
           [ [ 211, 212, 213, 214 ], [ 221, 222, 223, 224 ], [231, 232, 233, 234] ] ]

#%%
# Now try numpy
import numpy as np
# Some basic attributes and simple functions of numpy arrays
a = np.arange(15) # numpy creates a range of 15 consecutive integers, like the range() function in basic python
print('a:',a) 

#%%
a = np.arange(15).reshape(3,-1) # Using -1 for the last dimension lets numpy calculate directly
# a = np.arange(15).reshape(3,5) # Same result as line above
a = np.arange(24).reshape(2,3,-1) # 3d array
print('a:',a) 

print('a.shape:',a.shape) 
print('a.ndim:',a.ndim) 
print('a.dtype:',a.dtype) 
print('a.dtype.name:',a.dtype.name) 
print('a.itemsize:',a.itemsize) 
print('a.size:',a.size) 
print('type(a):',type(a)) 
b = np.array([6, 7, 8])
print('b:',(b)) 
print('type(b):',type(b)) 
#

#%%
# The opposite of reshape, can use ravel()
print('a ravel:', a.ravel().shape)
print('a again:', a) 
#
# IMPORTANT
# The a.ravel() function does NOT change a!! 
# I create a true copy of a and ravel/unravel it only. 
# Remember the differences in class/object definitions, 
# it is critical what is the "return" value in 
# those function/methods. 
# If return self, you are getting back the object a. 
# But this function return a separate true copy of 
# the result instead. This is by design. 
# 
# A lot of other functions in numpy/pandas behave like that too. 
# 
# The same thing for reshape, for example
print('a reshape:', a.reshape(1,-1))
print('a.shape:',a.shape) 
print('a:',a) 
print('#',50*"-")

#%%
# If you really want to change a, try this: 
# a = a.ravel() # exact same result as 
a = a.reshape(-1)
print('a: ',a)
print('type a: ',type(a))
print('a.shape: ',a.shape)

print('#',50*"-")

#%%
# Other examples to create some simply numpy arrays
print('zeros:', np.zeros( (3,4) ))
print('ones:', np.ones( (2,3,4), dtype=np.int16 ))
print('empty:', np.empty( (2,3) ))
print('arange variation 1:', np.arange( 10, 30, 5 ))
print('arange variation 2:', np.arange( 0, 2, 0.3 ) )
print('complex:', np.array( [ [1,2], [3,4] ], dtype=complex ))
print('float:', np.arange(2, 10, dtype=float) )
from numpy import pi
x = np.linspace( 0, 2*pi, 100 )
f = np.sin(x)
print('sine of linspace:',f)
print('#',50*"-")

#%%
# import numpy as np
# Creating numpy arrays from python lists (and other list-like objects)
# Also look at the concept of "strides"
nparray1 = np.array(list1)
print("nparray1 = \n", nparray1)
print("type(nparray1) =", type(nparray1))
print("nparray1.dtype =", nparray1.dtype)  # int64
print("nparray1.shape =", nparray1.shape)
print("nparray1.strides =", nparray1.strides)  # each value is int64, hence 8-byte of memory, with four columns, it takes 8x4 = 32 bytes to the next row, same position. Strides = (32,8) to the next row and next column


#%%
# if we redo 
nparray1 = np.array(list1, dtype= np.int32)
print("nparray1 = \n", nparray1)
print("type(nparray1) =", type(nparray1))
print("nparray1.dtype =", nparray1.dtype)  # int32
print("nparray1.shape =", nparray1.shape)
print("nparray1.strides =", nparray1.strides)  # now each value is int32, 4-byte, with four columns, it takes 4x4 = 16 bytes to next row. 


#%%
# Try others
nparray2 = np.array(list2)
print("nparray2 = \n", nparray2)
print("type(nparray2) =", type(nparray2))
print("nparray2.dtype =", nparray2.dtype)  # int64
print("nparray2.shape =", nparray2.shape)

#%%
import sys
try:
  nparray12 = nparray1+nparray2
except ValueError as err :  # except (RuntimeError, TypeError, NameError):
  print("Value Error: {0}".format(err), " Try transpose...")
  nparray12 = nparray1+nparray2.T
except TypeError as err :  # except (RuntimeError, TypeError, NameError):
  print("Type Error: {0}".format(err))
except:
  print("unexpected error:", sys.exc_info()[0])


#%%
list4 = [ 5, 'a', 2, 3.5, True ]
list5 = [ 5, [1,4], 3, 1 ]

nparray4 = np.array(list4)
print("nparray4 = \n", nparray4)
print("type(nparray4) =", type(nparray4))
print("nparray4.dtype =", nparray4.dtype)  
print("nparray4.shape =", nparray4.shape)

#%%
# list5 = [ 5, [1,4], 3, 1 ]
nparray5 = np.array(list5)
print("nparray5 = \n", nparray5)
print("type(nparray5) =", type(nparray5))
print("nparray5.dtype =", nparray5.dtype)  
print("nparray5.shape =", nparray5.shape)


#%%
# If they are 2D-arrays, and have compatible dimensions, you can multiply them as matrices
tprod12 = np.dot(nparray1,nparray2)
print("tprod12.shape =", tprod12.shape)
mprod21 = np.dot(nparray2,nparray1)
print("mprod21.shape =", mprod21.shape)


#%%
# Also try the 3d-array that we constructed...
# In physics, those are called tensors. 
nparray3D = np.array(list3D)
print("nparray3D = \n", nparray3D)
print("type(nparray3D) =", type(nparray3D))
print("nparray3D.dtype =", nparray3D.dtype)  
print("nparray3D.shape =", nparray3D.shape)

#%%
# If they are 2D-arrays, and have compatible dimensions, you can multiply them as matrices
tprod32 = np.dot(nparray3D,nparray2)
print("tprod32.shape =", tprod32.shape)


#%%[markdown]
# speed and ease of use is the strength of numpy array, compared to python lists. 
# The entire array must be of a single type, however.
# If we try to time or clock the code execution times, you will find similar functions 
# is much faster than looping thru a python list.
# This is mainly because NumPy is written in C, and optimized these specialized 
# operations in a well-designed library.

#%%
# filtering and indexing
print(nparray1[0:2,:2],'\n')
print(nparray1[:,-1:])

#%%
# Let us do something simpler.
# Obtain the third column of nparray1
print(nparray1)
v3 = nparray1[:,2]
print(v3) # it is a column vector, or array one by three (3,1)
print(v3.shape) # it is a column vector, or array one by three (3,1)

# Much easier than dealing with lists on the coding side of things. Speed is also maximized.


#%%
# BROADCASTING 
# 
# Let's practice slicing numpy arrays and using NumPy's broadcasting concept. 
# Remember, broadcasting refers to a numpy array's ability to VECTORIZE operations, 
# so they are performed on all elements of an object at once.
# If you need to perform some simple operations on all array elements, 
#
nparray1squared = nparray1 ** 2
print(nparray1squared)

#%%
nparray1mod7 = nparray1 % 7 # remainder from dividing by 7
print(nparray1mod7)

#%%
nparray1b = np.array(list1b)
nparray1bovera = nparray1b / nparray1
print(nparray1bovera)

# Try some other operations, see if they work.

# Next try to do the above with loops or comprehensions? 

#%%
# boolean indexing 
print(nparray1)
npbool1greater = nparray1 > 21
print(npbool1greater)

#%%
print(nparray1[npbool1greater])

#%%
print(nparray1[npbool1greater].shape)

#%%
npbool1mod = nparray1 %2 ==1
print(npbool1mod)
print(nparray1[npbool1mod])
print(nparray1[npbool1mod].shape)

# Again, try to do these with loops or comprehensions? 

#%%
# Let us look at filtering again. 
x = np.arange(10)
print('x:\n',x)
print('x[1]:',x[1])
print('x[-2]:',x[-2])
#%%
x.shape = (2, 5)
print('x[1,3]:',x[1, 3])
print('x[1,-1]:',x[1,-1])
print('x[0]:',x[0])
print('x[1][2]:',x[1][2])
print('x[1,2]:',x[1,2]) # SAME as above
print('x[2:5]:',x[2:5])
print('x[-7]:',x[:-7])
print('x[1:7:2]:',x[1:7:2])
y = np.arange(35).reshape(5,7)
print('\ny:\n',y)
print('y[1:5:2,::3]:',y[1:5:2,::3])
v = np.arange(10,1,-1)
print('\nv:\n',v)
print('v[np.array([3, 3, 1, 8])]:',v[np.array([3, 3, 1, 8])])
print('#',50*"-")
#%%
# Some other basic numpy operations
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print('b:',b)
c = a-b
print('c:',c)
print('b**2:',b**2)
print('sine:',10*np.sin(a))
print('#',50*"-")

#%%
a = np.random.random((2,3))
print('a:',a)
print('a.sum:',a.sum())
print('a.min:',a.min())
print('a.max:',a.max())
print('#',50*"-")
#%%
b = np.arange(24).reshape(3,4,2)
print('b:',b)
print('b.sum axis 0:',b.sum(axis=0)) # sum of each column
print('b.min axis 1:',b.min(axis=1)) # min of each row
print('b.max axis 2:',b.min(axis=2)) # min of each row
print('b.cumsum axis 1:',b.cumsum(axis=1)) # cumulative sum along each row
print('#',50*"-")

#%% 
# We looked at mostly the functions/methods within the numpy object, like 
# a.sum()
# a.min() etc.
# As we discussed in OOP class, a lot of times, we can also have methods for the class itself. 
# In this case, we can use the universal functions in the numpy library directly, 
# like np.exp(), np.sqrt(). np.add() etc.
a = np.arange(3)
print('a:',a)
e = np.exp(a); print('exp:',e)
rt = np.sqrt(a); print('root:',rt)
b = np.array([2., -1., 4.])
sum = np.add(a, b)
print('sum:',sum)
print('#',50*"-")

#%% 
# Indexing and slicing in Numpy
a = np.arange(10)**3; 
print('a:', a)
print('a[2]:', a[2])
print('a[2:5]:', a[2:5]) # same as basic python syntax, not including 5.
a[:6:2] = -1000
print('a[::-1] :', a[ : :-1])
# reset a
a = np.arange(10)**3
for i in a:
  print('i-cube rt:', np.power(i,1/3))
#
#%%
# def f(i,j):
#   return 10*i+j
# b = np.fromfunction(f,(5,4),dtype=int); # The array values for point/coordinate (i,j,k) will be f(i,j,k). 
# 
# OR
# use the lambda function
b = np.fromfunction(lambda i,j: 10*i+j,(5,4),dtype=int); # The array values for point/coordinate (i,j,k) will be f(i,j,k). 
print('b:', b)
print('b[2,3]:', b[2,3])
print('b[0.5,1]:', b[0:5, 1])
print('b[:,1]:', b[ : ,1])
print('b[1:3, :]:', b[1:3, : ])
print('b[-1]:', b[-1])  # same as b[4]  # same as b[4, : ] 
print('b[4]:', b[4]) 
print('b[4,:]:', b[4, :]) 
print('#',50*"-")
#
# There is also a np.fromfile() function  to create.
#
# Remeber we tried slicing of 2-D list/array with basic python 
# at the beginning of this file, and it was super difficult to 
# get what we need. 

#%%



