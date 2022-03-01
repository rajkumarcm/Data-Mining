#%%
# print("Hello world!")


#%%
# Question 1: Create a Markdown cell with the followings:
# Two paragraphs about yourself. In one of the paragraphs, give a hyperlink of a website 
# that you want us to see. Can be about yourself, or something you like.

#%% [markdown]
###### My name is Rajkumar Conjeevaram Mohan, who graduated in Artificial Intelligence from Imperial College London also happens to be a technology enthusiast. My coding journey began when I was in the 6th grade, and HTML was the first web development language I was exposed to. I slowly started studying C, and by now, I have already learnt and used Java, C, C++, Python, MATLAB, .Net, PHP, Prolog, Oracle. Also, I obtained a diploma in Computer Hardware when I was below 17 years old and love troubleshooting computers. Most of my summer holidays were spent in some institute learning some computer related work – I have got my hands on Adobe Multimedia Flash 5 to create animations, and I truly take the pleasure in writing complex code that probably many do not even know about. One of the languages that I have always admired at, would be Prolog as I happened to have liked the syntax and it would be my understand that logic programming would be useful at some time for my research.
###### It would be my long-term goal to pursue doctoral research in Robotics, which is one of the reasons why I push myself hard in learning things that are relevant, though outside the University course syllabus. Though not an expert now, I have the confidence that I will reach the apex at some point. I believe my LinkedIn profile that can be accessed through https://www.linkedin.com/in/rajkumarcm/ can talk about the rest.
######


#%%
# Question 2: Create
# a list of all the class titles that you are planning to take in the data science program. 
# Have at least 6 classes, even if you are not a DS major
# Then print out the last entry in your list.
class_titles = ['Data Warehousing', 'Intro to Data Science', 'Intro to Data Mining', \
                'Machine Learning I', 'Data Visualization', 'Machine Learning II']
print(class_titles[-1])

#%%
# Question 3: After you completed question 2, you feel Intro to data mining is too stupid, so you are going 
# to replace it with Intro to Coal mining. Do that in python here.
class_titles[2] = 'Intro to Coal Mining'


#%%
# Question 4: Before you go see your acadmic advisor, you are 
# asked to create a python dictionary of the classes you plan to take, 
# with the course number as key. Please do that. Don't forget that your advisor 
# probably doesn't like coal. And that coal mining class doesn't even have a 
# course number.

# There are two ways to do this: I will probably do the fancier way.
# course_dict = {6102: 'Data Warehousing', 6101: 'Intro to Data Science', 6103: 'Intro to Data Mining',
#                6202: 'Machine Learning I', 6401: 'Data Visualization', 6203: 'Machine Learning II'}
course_code = [6102, 6101, 6103, 6202, 6401, 6203]
class_titles[2] = 'Intro to Data Mining' # I don't do anything my advisor doesn't like
course_dict = {key:value for key, value in zip(course_code, class_titles)}

#%%
# Question 5: print out and show your advisor how many 
# classes (print out the number, not the list/dictionary) you plan 
# to take.
n = len(course_dict.items())
# n = len(class_titles)
print(f"I wish to take {n} courses for the first two semesters")
#%%
# Question 6: Using loops 
# DO NOT use any datetime library in this exercise here 
# Use only basic loops
# Goal: print out the list of days (31) in Jan 2022 like this
# Sat - 2022/1/1
# Sun - 2022/1/2
# Mon - 2022/1/3
# Tue - 2022/1/4
# Wed - 2022/1/5
# Thu - 2022/1/6
# Fri - 2022/1/7
# Sat - 2022/1/8
# Sun - 2022/1/9
# Mon - 2022/1/10
# Tue - 2022/1/11
# Wed - 2022/1/12
# Thu - 2022/1/13
# ...
# You might find something like this useful, especially if you use the remainder property x%7
dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat') # day-of-week-tuple
for i in range(1, 32):
    day_in_week = ((i-1) % 7) - 1 # I had to do -1 since it starts from Saturday and Saturday is the last element in the list
    day_abbrv = dayofweektuple[day_in_week]
    print(f"{day_abbrv} - 2022/1/{i}")


# %%[markdown]
# # Additional Exercise: 
# Choose three of the four exercises below to complete.
#%%
# =================================================================
# Class_Ex1: 
# Write python codes that converts seconds, say 257364 seconds,  to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------
def convert_seconds(seconds):
    hour = seconds//3600
    minutes = (seconds - hour * 3600)//60
    second = seconds - (hour * 3600 + minutes * 60)
    return (hour, minutes, second)

hour, mins, secs = convert_seconds(257364)
print(f"Result: {hour} hr {mins} mins {secs} secs")

""" Personal test cases
# Test 1
# 1 hour 10 mins 25 seconds
total_seconds = 3600 + 10*60 + 25
hour, mins, secs = convert_seconds(total_seconds)
print(f"Expected: 1 hour 10 mins 25 secs\nObserved: {hour} hour {mins} mins {secs} secs\n")

# Test 2
# 2 hour 05 mins 15 seconds
total_seconds = 7200 + 5*60 + 15
hour, mins, secs = convert_seconds(total_seconds)
print(f"Expected: 2 hour 5 mins 15 secs\nObserved: {hour} hour {mins} mins {secs} secs")
"""


#%%
# =================================================================
# Class_Ex2: 
# Write a python codes to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# Hint: one way is to create three nested loops.
# ----------------------------------------------------------------
letters = ['A', 'B', 'C']
for p in letters:
    for q in letters:
        for r in letters:
            if p != q and p != r and q != r:
                print(f"{p},{q},{r}")




#%%
# =================================================================
# Class_Ex3: 
# Write a python codes to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------
letters = ['A', 'B', 'C', 'D']
for p in letters:
    for q in letters:
        for r in letters:
            for s in letters:
                if p != q and p != r and p != s and q != r and q != s and r != s:
                    print(f"{p},{q},{r},{s}")



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
height = 5
for i in range(height):
    for _ in range(height - i - 1):
        print(" ", end="")
    if i == 0:
        print("*")
        reps = 1
    else:
        reps += 2  # whatever the previous length is, add one to the left and one to the right
        print("*"*reps)



# %%
