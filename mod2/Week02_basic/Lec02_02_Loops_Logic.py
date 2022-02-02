#%%
import math 
import os 


print("Hello world!")

#%%
# From before
abool = True    # boolean
azero = 0    # int
aint = 35    # int
afloat = -2.8     # float
anumbzerostr = "0"     # str
aemptystr = ""     # str
aletter = 'a'     # str
astr = "three-five"     # str
# list / array
alist = [1,'person',1,'heart',10,'fingers']
# tuple # like list, but immutable (faster access and processing)
atuple = (1,'person',1,'heart',10,'fingers')
# set  # un-ordered, and distinct elements.  
aset = { 1,'person',1,'heart',10,'fingers' }
# dictionary
adictionary = { "name": "Einstein", 1: "one", astr: 35, aint: 'thirty five', "last": alist }

#%%
# some more 
# note anything unexpected/unusual
list1 = [1,5,3,8,2]
list2 = [2]
tuple1 = (1,5,3,8,2)
print("type of tuple1: %s, length of tuple1: %d" % (type(tuple1), len(tuple1)) )

tuple2 = (2)
print("type of tuple2: %s" % type(tuple2) )
# print("type of tuple2: %s, length of tuple2: %d" % (type(tuple2), len(tuple2)) )
# len(tuple2) # does not work, error

tuple3 = tuple([2])
print("type of tuple3: %s, length of tuple3: %d" % (type(tuple3), len(tuple3)) )

tuple4 = ()
print("type of tuple4: %s, length of tuple4: %d" % (type(tuple4), len(tuple4)) )


#%%
# Slicing parts of list/tuple/set
# Try
# write some notes/comments for each case, so that you can review them easily yourself
print(alist[1:4])  # inclusive on the start index, exclusive of the end index
print(alist[:4])
print(alist[:])
# optional argument, skipping every 1 element with :2 at the end
print(alist[1:4:2])
print(alist[1:5:2])
print(alist[1:3:2])
# what do you expect the result of this to be?
print(alist[1::2])
# Also try )
print(alist[-4])
print(alist[-4:-2])
print(alist[-4:])
print(alist[-2:-4])

#%%
# Now try tuple, set, and dictionary
# Put some notes for yourself
# comment out the illegal ones so that you can run your entire file gracefully

#%%[markdown]
# # Logic
# ## Conditional statment
# 
# _________________________________________________  
# Statement:     If p, then q     OR   p  ->  q   
#
# Contrapositve: If -q, then -p   OR   -q -> -p  
# _________________________________________________  
# Inverse:       If -p, then -q   OR   -p ->  -q   
# 
# Converse:      If q, then p     OR   q  ->  p  
# _________________________________________________  

#%%[markdown]
# ## Some other logic rules
# 
# _________________________________________________  
# -(p AND q)
#
# same as 
#
# -p OR -q
# _________________________________________________  
# -(p OR q)
#
# same as 
#
# -p AND -q
# _________________________________________________  
# p AND (q AND r)
#
# same as 
#
# (p AND q) AND r
#
# we usually combine as 
#
# (p AND q AND r)
# _________________________________________________  
# p OR (q OR r)
#
# same as 
#
# (p OR q) OR r
#
# we usually combine as 
#
# (p OR q OR r)
# _________________________________________________  
# ## Distributive law 1
# p AND (q OR r)
#
# same as 
#
# (p AND q) OR (p AND r)
# _________________________________________________  
# ## Distributive law 2
# p OR (q AND r)
#
# same as 
#
# (p OR q) AND (p OR r)
# _________________________________________________  
#
#

#%%
# Basic logic
x = 1
y = 2
b = (x == 1)
b = (x != 1)
b = (x == 1 and y == 2)
b = (x != 1 and y == 2)
b = (x == 1 and y != 2)
b = (x != 1 and y != 2)
b = (x == 1 or y == 2)
b = (x != 1 or y == 2)
b = (x == 1 or y != 2)
b = (x != 1 or y != 2)
if x == 1 or 2 or 3:
	print("OK")
x == 1 or 2 or 3


#%%
# conditional
# if :
income = 60000
if income >100000 :
  print("rich")
# if else:
if income >100000 :
  print("rich")
else :
  print("not rich")
# if elif elif .... :
if income >200000 :
  print("super rich")
elif income > 100000 :
  print("rich")
elif income > 40000 :
  print("not bad")
elif income > 0 :
  print("could be better")
else :
  print("no idea")

# The above can be compacted into a one-liner
print("super rich" if income > 200000 else "rich" if income > 100000 else "not bad" if income > 40000 else "could be better" if income > 0 else "no idea" )
# or 
incomelevel = "super rich" if income > 200000 else "rich" if income > 100000 else "not bad" if income > 40000 else "could be better" if income > 0 else "no idea" 
print(incomelevel)

# write your conditional statment to assign letter grades A, A-, B+ etc according to the syllabus

#%%
# loops - basic
for i in range(10):
  print(i)

#%%
# loops - basic
print("looping i:")
for i in range(1000):
  print('still going',i)
  if i>13:
    break

#%%
print("\nlooping j:")
for j in range(2,1000,2):
  if j<933:
    continue
  print(j)
  if j>945:
    break
  # if j<938: # Try setting it to 938 or 958, see the difference
  if j<958: # Try setting it to 938 or 958, see the difference
    continue
  print("Can you see me?")

#%%
# loops - iterate a list/tuple/set/dictionary
# any difference among the three below?

# for val in list :
print("\nloop thru val in list:")
for val in [ 4,'2',("a",5),'end' ] :
  print(val, type(val))

# for val in tuple :
print("\nloop thru val in tuple:")
for val in ( 4,'2',("a",5),'end' ) :
  print(val, type(val))

# for val in set :
print("\nloop thru val in set:")
for val in { 4,'2',("a",5),'end' } :
  print(val, type(val))

#%%
# Now for dictionary
# for val in dictionary : (keys only)
print("\nloop thru key, val in dictionary??")
adictionary = { "k0":4, "k8":'2', "k1":("a",5), "k5":'end' }
for key in adictionary :
  print('key:', key, '; val', adictionary[key])

#%%
# or try this for dictionary, using .items() to get the pairs
print("\nalternative method to loop thru key, val in dictionary:")
# adictionary.items() # creates a object type of dict_items, which can be looped thru as key/value pairs   
for key, val in adictionary.items() :
  # print("key:", key, "value:", val, "type of value", type(val))
  print(f"key: {key}, value: {val}, and type of val: {type(val)}" )

#%%
# for val in string :
print("\nloop characters in a string:")
for char in 'GW Rocks' :
  print(char, type(char))
  
  
#%%
# Use enumerate function to generate the index??
# for index, val in enumerate(list) :
print("\nloop index value pairs in a list:")
alist = [ 4,'2',("a",5),'end' ]
for index, val in enumerate(alist) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print()

#%%
# Try tuple, set, and dictionary
print("\nloop key value pairs in a tuple like that?")
atuple = ( 4,'2',("a",5),'end' )
for index, val in enumerate(atuple) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("OK\n")

print("\nloop key value pairs in a set like that?")
aset = { 4,'2',("a",5),'end' }
for index, val in enumerate(aset) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("OK. BUT order is messed up!!\n")

#%%
print("\nloop key value pairs in a dictionary like that?")
adictionary = { "k0":4, "k8":'2', "k1":("a",5), "k5":'end' }
for index, val in enumerate(adictionary) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("Not quite what we want!!\n")


# %%
