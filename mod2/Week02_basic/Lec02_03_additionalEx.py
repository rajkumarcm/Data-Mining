# Exercises from Prof Amir Jafari

#%%
# =================================================================
# Class_Ex1: 
# Write python program that converts seconds to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------
def convert_seconds(seconds):
    hour = seconds//3600
    minutes = (seconds - hour * 3600)//60
    second = seconds - (hour * 3600 + minutes * 60)
    return (hour, minutes, second)

# Test 1---------------------------------------------------------------------------------
# 1 hour 10 mins 25 seconds
total_seconds = 3600 + 10*60 + 25
hour, mins, secs = convert_seconds(total_seconds)
print(f"Expected: 1 hour 10 mins 25 secs\nObserved: {hour} hour {mins} mins {secs} secs\n")

# Test 2---------------------------------------------------------------------------------
# 2 hour 05 mins 15 seconds
total_seconds = 7200 + 5*60 + 15
hour, mins, secs = convert_seconds(total_seconds)
print(f"Expected: 2 hour 5 mins 15 secs\nObserved: {hour} hour {mins} mins {secs} secs")
print('debug...')
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






# =================================================================