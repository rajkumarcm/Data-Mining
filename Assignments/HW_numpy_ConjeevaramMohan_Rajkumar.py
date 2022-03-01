# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# HW Numpy 
# ## By: Rajkumar Conjeevaram Mohan
# ### Date: 20th February 2022
#


#%%
# NumPy

import numpy as np

# %%
# ######  QUESTION 1      QUESTION 1      QUESTION 1   ##########
# This exercise is to test true/shallow copies, and related concepts. 
# ----------------------------------------------------------------
# 
# ######  Part 1a      Part 1a      Part 1a   ##########
# 
list2 = [ [11,12,13], [21,22,23], [31,32,33], [41,42,43] ] # two dimensional list (2-D array)  # (4,3)
nparray2 = np.array(list2)
print("nparray2:\n", nparray2)

# We will explain more of this indices function in class next. See notes in Class05_02_NumpyCont.py
idlabels = np.indices( (4,3) ) 
print("idlabels:\n", idlabels)

i,j = idlabels  # idlabels is a tuple of length 2. We'll call those i and j
# DEBUG..............................................


nparray2b = 10*i+j+11
print("nparray2b:\n",nparray2b)

# 1.a) Is nparray2 and nparray2b the "same"? Use the logical "==" test and the "is" test. 
# Write your codes, 
# and describe what you find.
res = (list2 == nparray2b).all()
print(f"Does list2 has elements identical to nparray2b?: {res}")
# Obviously it will return False as they both do not refer to the same address in memory
res = list2 is nparray2b
print(f"Although list2 has elements identical to nparray2b, they both do not point to the same address in memory. \
Hence the result for is test would be: {res}")
#%% [markdown]
# Yes, they both are the same. np.indices return grid of indices row and columns as tuples.
# From my understanding i*10+11 makes row indices scaled by 10 and shifted by 11 i.e., each row represents the offset for elements that are 10 indices apart starting from 11
# and the remaining j adds column indices to row offset that altogether means retrieving first three elements for each 10 elements starting from 11.

# %%
# ######  Part 1b      Part 1b      Part 1b   ##########
# 
# 1.b) What kind of object is i, j, and idlabels? Their shapes? Data types? Strides?
# 
# write your codes here

data_type = lambda x: "ndarray" if type(x) is np.ndarray else type(x)

print(f"i: type: {data_type(i)}, shape={i.shape}, dtype={i.dtype}, strides={i.strides}")
print(f"j: type: {data_type(j)}, shape={j.shape}, dtype={j.dtype}, strides={j.strides}")
print(f"idlabels: type: {data_type(idlabels)}, shape={idlabels.shape}, datatype={idlabels.dtype}, strides={idlabels.strides}")


# %%
# ######  Part 1c      Part 1c      Part 1c   ##########
# 
# 1.c) If you change the value of i[0,0] to, say 8, print out the values for i and idlabels, both 
# before and after the change.
# write your codes here
i[0,0] = 8
print("i:\n", i)
print("idlabels:\n", idlabels)
# Describe what you find. Is that what you expect?
#%% [markdown]
# I understand that i[0,0] = 8 is a shallow copy, which only copies memory address of the original object
# so that any changes to the copied object will also reflect on the original object as they both are two
# different objects with reference to the same memory address.  

#%%
# Also try to change i[0] = 8. Print out the i and idlabels again.
i[0] = 8
print(f"After changing i[0]=8")
print("i:\n", i)
print("idlabels:\n", idlabels)
#%% [markdown]
# i[0]=8 is a broadcast process that assigns 8 to all elements in the first row of i and this will reflect also in idlabels


# %%
# ######  Part 1d      Part 1d      Part 1d   ##########
# 
# 1.d) Let us focus on nparray2 now. (It has the same values as nparray2b.) 
# Make a shallow copy nparray2 as nparray2c
# now change nparray2c 1,1 position to 0. Check nparray2 and nparray2c again. 
# Print out the two arrays now. Is that what you expect?
nparray2c = nparray2
nparray2c[1,1]=0
print(f"nparray2:\n{nparray2}")
print(f"\nnparray2c:\n{nparray2c}")
# 
# Also use the "==" operator and "is" operator to test the 2 arrays. 
# write your codes here
res = (nparray2 == nparray2c).all()
print(f"Are all the elements in nparray2c identical to nparray2 even after nparray2c[1,1]=0 ?: {res}")
res = (nparray2 is nparray2c)
print(f"Is nparray2c the same as nparray2 ?: {res}. This is because we did a shallow copy of nparray2 to nparray2c that only copied the memory address to nparray2c")

#%%
# ######  Part 1e      Part 1e      Part 1e   ##########
# Let us try again this time using the intrinsic .copy() function of numpy array objects. 
nparray2 = np.array(list2) # reset the values. list2 was never changed.
nparray2c = nparray2.copy() 
# now change nparray2c 0,2 position value to -1. Check nparray2 and nparray2c again.
nparray2c[0, 2] = -1
print(f"nparray2:\n{nparray2}\n")
print(f"nparray2c:\n{nparray2c}\n")

#%% [markdown]

# Are they true copies?\
#\
# Yes, nparray2 and nparray2c are true copies each having their own memory reference.
#%%
# write your codes here
# Again use the "==" operator and "is" operator to test the 2 arrays. 
res = (nparray2c == nparray2).all()
print(f"Perhaps this is a deep copy, which creates a new memory address with all elements being recursively copied. \
This would make both the arrays possess the identical elements (before reassigning new values), but not passing the \
is condition as they both are two different objects with different memory addresses. The answer to the verification \
whether nparray2 and nparray2c have same elements: {res} as changes to nparray2c will not affect nparray2")
res = nparray2 is nparray2c
print(f"\nIs nparray2c the same as nparray2?: {res}")
# Since numpy can only have an array with all values of the same type, we usually 
# do not need to worry about deep levels copying. 
# 
# ######  END of QUESTION 1    ###   END of QUESTION 1   ##########




# %%
# ######  QUESTION 2      QUESTION 2      QUESTION 2   ##########
# Write NumPy code to test if two arrays are element-wise equal
# within a (standard) tolerance.
# between the pairs of arrays/lists: [1e10,1e-7] and [1.00001e10,1e-8]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.00001e10,1e-9]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.0001e10,1e-9]
# Try just google what function to use to test numpy arrays within a tolerance.
print(np.allclose([1e10,1e-7], [1.00001e10,1e-8]))
print(np.allclose([1e10,1e-8], [1.00001e10,1e-9]))
print(np.allclose([1e10,1e-8], [1.0001e10,1e-9]))
print(np.allclose([1e-2], [1e-2+1e-7])) # My own experimental case
print(f"Standard tolerance appears to be t <= 1e-7")

# ######  END of QUESTION 2    ###   END of QUESTION 2   ##########


# %%
# ######  QUESTION 3      QUESTION 3      QUESTION 3   ##########
# Write NumPy code to reverse (flip) an array (first element becomes last).
x = np.arange(12, 38)
print(f"Reverse x:\n{x[::-1]}\n")

# ######  END of QUESTION 3    ###   END of QUESTION 3   ##########


# %%
# ######  QUESTION 4      QUESTION 4      QUESTION 4   ##########
# First write NumPy code to create an 7x7 array with ones.
# Then change all the "inside" ones to zeros. (Leave the first 
# and last rows untouched, for all other rows, the first and last 
# values untouched.) 
# This way, when the array is finalized and printe out, it looks like 
# a square boundary with ones, and all zeros inside. 
# ----------------------------------------------------------------
tmp_ones = np.ones([7,7])
tmp_ones[1:-1, 1:-1] = 0
print(tmp_ones)

# ######  END of QUESTION 4    ###   END of QUESTION 4   ##########



# %%
# ######  QUESTION 5      QUESTION 5      QUESTION 5   ##########
# Broadcasting, Boolean arrays and Boolean indexing.
# ----------------------------------------------------------------
i=3642
myarray = np.arange(i,i+6*11).reshape(6,11)
print(myarray)
# 
# a) Obtain a boolean matrix of the same dimension, indicating if 
# the value is divisible by 7. 
div_by_7 = myarray%7 == 0
#print(f"\nBoolean matrix indicating elements divisible by 7:\n{div_by_7}\n")

# b) Next get the list/array of those values of multiples of 7 in that original array  
m_of_7 = myarray[div_by_7]
print(f"\nMultiples of 7:\n{m_of_7}\n")
# ######  END of QUESTION 5    ###   END of QUESTION 5   ##########





#
# The following exercises are  
# from https://www.machinelearningplus.com/python/101-numpy-exercises-python/ 
# and https://www.w3resource.com/python-exercises/numpy/index-array.php
# Complete the following tasks
# 

# ######  QUESTION 6      QUESTION 6      QUESTION 6   ##########

#%%
flatlist = list(range(1,25))
print(flatlist) 

#%%
# 6.1) create a numpy array from flatlist, call it nparray1. What is the shape of nparray1?
# remember to print the result
#
# write your codes here
nparray1 = np.array(flatlist)
print(f"nparray1.shape: {nparray1.shape}")

#%%
# 6.2) reshape nparray1 into a 3x8 numpy array, call it nparray2
# remember to print the result
#
# write your codes here
nparray2 = nparray1.reshape([3, 8])
print(f"\nnparray2:\n{nparray2}\n")

#%%
# 6.3) swap columns 0 and 2 of nparray2, and call it nparray3
# remember to print the result
#
# write your codes here
indices = list(range(nparray2.shape[1]))
indices[0] = 2
indices[2] = 0
nparray3 = nparray2[:, indices]
print(f"\nnparray3:\n{nparray3}\n")

#%%
# 6.4) swap rows 0 and 1 of nparray3, and call it nparray4
# remember to print the result
# write your codes here
indices = list(range(nparray3.shape[0]))
indices[0] = 1
indices[1] = 0
nparray4 = nparray3[indices]
print(f"\nnparray4:\n{nparray4}\n")

#%%
# 6.5) reshape nparray4 into a 2x3x4 numpy array, call it nparray3D
# remember to print the result
#
# write your codes here
nparray3D = nparray4.reshape([-1, 3, 4])
print(f"\nnparray3D:\n{nparray3D}\n")

#%%
# 6.6) from nparray3D, create a numpy array with boolean values True/False, whether 
# the value is a multiple of three. Call this nparray5
# remember to print the result
# 
# write your codes here
nparray5 = nparray3D%3 == 0
print(f"\nnparray5:\n{nparray5}\n")

#%%
# 6.7) from nparray5 and nparray3D, filter out the elements that are divisible 
# by 3, and save it as nparray6a. What is the shape of nparray6a?
# remember to print the result
#
# write your codes here
nparray6a = nparray3D[nparray5]
print(f"\nnparray6a shape: {nparray6a.shape}:\n{nparray6a}\n")

#%%
# 6.8) Instead of getting a flat array structure, can you try to perform the filtering 
# in 6.7, but resulting in a numpy array the same shape as nparray3D? Say if a number 
# is divisible by 3, keep it. If not, replace by zero. Try.
# Save the result as nparray6b
# remember to print the result
# 
# write your codes here
nparray6b = nparray3D * nparray5
print(f"\nnparray6b:\n{nparray6b}")
# 
# ######  END of QUESTION 6    ###   END of QUESTION 6   ##########

#%%
#
