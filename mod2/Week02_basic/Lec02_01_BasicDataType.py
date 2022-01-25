#%%
print("Hello world!")
print(5 / 8)
print (7+10)
print(10/3, 3/10, 10//3, 3//10)    # regular division, regular division, quotient, quotient
print(10%3, 3%10)     #  remainder, which is the complement of quotient.  (This is also often referred as "modular arithmetic" or "modular algebra")

#%%[markdown]
#
# # Python Week 02
# 
# Above was an h1 header.
# Hello to everyone.   
#
# This can get you a [link](http://www.gwu.edu).
#
# You can find some cheatsheets to do other basic stuff like bold-face, italicize, tables, etc.

#%%
# # Four basic data type in Python (compared to Java/C++, python is very limited that way, for good and bad)
# BIFS - boolean, integer, float, string

abool = True    # boolean
aboolstr = "True"     # str
azero = 0    # int
aint = 35    # int
aintstr = "35"     # str
afloat = -2.8     # float
afloatstr = "-2.8"     # str
anumbzerostr = "0"     # str
aemptystr = ""     # str
aletter = 'a'     # str
astr = "three-five"     # str

# %%
# First, let us try a little interactive - allow user input to set a parameter
userin = input("What is your name?")
print(f'Hello {userin}\n')

# %%
userin = input("What is your favorite integer?")
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')

# %%
# TRY AGAIN
userin = int(input("What is your favorite integer?"))
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')

# OR
# Look up the eval() function in python, see what it does. 
userin = eval(input("What is your favorite integer?"))
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')




#%%
# higher level data types (class)
# list / array
alist = [1,'person',1,'heart',10,'fingers']

# tuple # like list, but an immutable object (faster access and processing)
atuple = (1,'person',1,'heart',10,'fingers')

# set  # un-ordered, and distinct elements.  
aset = { 1,'person',1,'heart',10,'fingers' }


#%%
# dictionary # like associative array in other lang.  
# The list is not indexed (by integers), but reference by a key.
# #######################################
# The key must be a primitive data type 
# preferrably use only Int and Str !!!
# #######################################
adictionary = { "name": "Einstein", 1: "one", "love": 3.14159 }
# access elements with 
adictionary['love']

#%%
# This kind of works too?! 
# Also kind of strange to use float and bool as key though, but it is possible and you might find it reasonable in certain situations.
adictionary2 = { "name": "Einstein", 1: "one", 3.14: 3.14159, True: 'love', "last": alist }
print(adictionary2)
print(type(adictionary2["last"]))
print(len(adictionary2))

#%%
# ######## BUT BE VERY CAREFUL if you use bool and float for keys. They might not be what you expect.
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))
adictionary4 = { "name": "Einstein", 2: "two", 2.0: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))
# below does not work. you can try by uncommenting it and run the line code
# notadicationary = { ['a',2]: "sorry", {1,2}: ('not','ok') }

#%%
# ###################  1. Exercise    Exercise    Exercise   ################################## 
# Try to create some more complicated entities. List of tuples, dictionary of dictionary, see if you find anything unexpected. 
print("This is exercise 1")




#%%
# ###################  2. Exercise    Exercise    Exercise   ################################## 
# Implicit conversion, which is also called coercion, is automatically done. (different lang has different coercion rules.)
# Explicit conversion, which is also called casting, is performed by code instructions.
print("This is exercise 2")


#%%
# Example, try 
int(abool)
str(abool)
str(int(abool))
# int(str(abool))

#%%
# Try it yourself, using the functions bool(), int(), float(), str() to convert. 
# what are the ones that you surprises you? List them out for your own sake




#%% 
# ####################  3. Exercise - binary operations:  ################################## 
# try to add or multiply differnt combinations and see the result. 
# Show your work here
print("This is exercise 3")

# Example -- implicit conversion is automatic here
add_bool_zero = abool + azero
print('result: type= ' + str(type(add_bool_zero)) + ' , value= ' +str(add_bool_zero) )



#%%
# ####################  4. Exercise - copying/cloning/deep cloning/shallow copy  ################################## 
# copy vs reference 
print("This is exercise 4")
cbool = abool
abool = False
print(abool)
print(cbool)
#do the same for the four differnt types


#%%
# ####################  Next, try it on tuple, list, set, dictionary ####################
ctuple = atuple
ctuple = (1,'person','2','hearts', 6 , 'fingers')
print(atuple)
print(ctuple)
# notice that tuples cannot assign a new value individually like atuple[1]='guy', but you can reassign the entire variable
clist = list(alist)
clist = alist[:]
# clist = alist
clist[2]=2
clist[3] = 'hands'
print(alist)
print(clist)
# Is it what you expect??

#%%
# Now try the other data types: set, dictionary, set of dictionaries, list of tuples, 
# etc etc
# These are shallow copies. They just copy the reference address, not the (primitive) values. 
# How do we make static clones that are no longer tied?
# Try google
# Does that work for deep level objects like list of dictionaries?
#
print(alist) # check the values for alist
# reset adictionary3
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }
# len(adictionary3)
acopy1 = adictionary3
acopy2 = adictionary3.copy()

import copy
acopy3 = copy.copy(adictionary3)

acopy1[2] = 'duo'

print(adictionary3)
print(acopy1)
print(acopy2)
print(acopy3)

 
#%%
# Let us get some help from the package "copy"
#
# reset alist
alist = [1, 'person', 1, 'heart', 10, 'fingers']
# reset adictionary3
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }

# len(adictionary3)
acopy1 = adictionary3
acopy2 = adictionary3.copy()

import copy
acopy3 = copy.copy(adictionary3)
acopy4 = copy.deepcopy(adictionary3)

acopy1[2] = 'duo'
acopy1['last'][3]='nose'

print(adictionary3)
print(acopy1)
print(acopy2)
print(acopy3)
print(acopy4)


# %%[markdown]
# The copy.deepcopy() method works! It works recursively on lists/dictionary, etc, such as JSON objects. 
# Needless to say, use it only if it is needed, as it costs performance-wise. It also might not work if 
# the object type is other more complicated objects.
#

#%%
