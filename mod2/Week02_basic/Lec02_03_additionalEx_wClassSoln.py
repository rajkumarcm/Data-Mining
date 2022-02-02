# Exercises from Prof Amir Jafari

#%%
# =================================================================
# Class_Ex1: 
# Write python program that converts seconds to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------





#%%
# =================================================================
# Class_Ex2: 
# Write a python program to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# ----------------------------------------------------------------





#%%
# =================================================================
# Class_Ex3: 
# Write a python program to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------





#%%
# =================================================================
# Class_Ex4: 
# Suppose we wish to draw a triangular tree, and its height is provided 
# by the user, like this, for a height of 5:
#      *
#     ***
#    *****
#   *******
#  *********
# ----------------------------------------------------------------





#%%
# =================================================================
# Class_Ex5: 
# Write python program to print prime numbers up to a specified values.
# ----------------------------------------------------------------

nmax =200
for i in range (2,nmax+1):
  chk = True
  for j in range(2,i):
    if i%j !=0 : 
      continue
    else: 
      chk = False
      break
  if(chk): print(i)


#%%
prime = []
nmax = 200
for i in range(2, nmax + 1):
    rem = []
    for j in range(2, 15):
        if j <= i and i != j:
            rem = i % j
            if rem == 0:
                break
    if rem != 0:
        prime.append(i)
print(prime)



# =================================================================
# %%
