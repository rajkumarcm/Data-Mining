###############  HW  Functions      HW  Functions         HW  Functions       ###############
#%%
# ######  QUESTION 1   First, review Looping    ##########
# Write python codes to print out the four academic years for a typical undergrad will spend here at GW. 
# Starts with Sept 2021, ending with May 2025 (total of 45 months), with printout like this:
# Sept 2021
# Oct 2021
# Nov 2021
# ...
# ...
# Apr 2025
# May 2025
# This might be helpful:
# If you consider Sept 2021 as a number 2021 + 8/12, you can continue to loop the increament easily 
# and get the desired year and month. (If the system messes up a month or two because of rounding, 
# that's okay for this exercise).
# And use this (copy and paste) 
from calendar import month


monthofyear = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec')
# to simplify your codes.
    
start_month_idx = 8
end_month_idx = 4
for y in range(2021, 2026):
  start = start_month_idx if y == 2021 else 0
  end = end_month_idx+1 if y == 2025 else len(monthofyear)
  for m in range(start, end):
    print(f"{monthofyear[m]} {y}")


###############  Now:     Functions          Functions             Functions       ###############
# We will now continue to complete the grade record that we were working on in class.

#%%
###################################### Question 2 ###############################
# let us write a function find_grade(total) 
# which will take your course total (0-100), and output the letter grade (see your syllabus)
# have a habbit of putting in the docstring
total = 62.1

def find_grade(total):
  # write an appropriate and helpful docstring
  # ??????    fill in your codes here, be sure you have all A, A-, ... thru D, and F grades completed.
  # grade = ???
    """
    Find the grade letter based on the total mark

    :param total: Float value that represent the total mark
    :return: String value that represent the grade letter
    """
  
    total = round(total)
    # I personally believe the underlying code is better than comprehensive style
    # as this is more readable especially when there are many cascading conditions.
    # grade = None
    # if total >= 93:
    #   grade = "A"
    # elif total >= 90:
    #   grade = "A-"
    # elif total >= 87:
    #   grade = "B+"
    # elif total >= 83:
    #   grade = "B"
    # elif total >= 80:
    #   grade = "B-"
    # elif total >= 77:
    #   grade = "C+"
    # elif total >= 73:
    #   grade = "C"
    # elif total >= 70:
    #   grade = "C-"
    # elif total >= 60:
    #   grade = "D"
    # else:
    #   grade = "F"

    # Just in case you want to see this here it is.
    grade = "A" if total >= 93 else "A-" if total >= 90 else "B+" if total >= 87 else "B" \
             if total >= 83 else "B-" if total >= 80 else "C+" if total >= 77 else "C" \
               if total >= 73 else "C-" if total >= 70 else "D" if total >= 60 else "F"

    return grade
# Try:
print(find_grade(total))

# Also answer these: 
# What is the input (function argument) data type for total? 
# What is the output (function return) data type for find_grade(total) ?
#%% [markdown]
# The input variable total is of data type float, and it returns a String value.

#%%
###################################### Question 3 ###############################
# next the function to_gradepoint(grade)
# which convert a letter grade to a grade point. A is 4.0, A- is 3.7, etc
grade = 'C-'

def to_gradepoint(grade):
  """
  Convert letter grade to grade point

  :param grade: String that represent the letter grade
  :return: Float value that represent the grade point
  """
  # write an appropriate and helpful docstring
  # ??????    fill in your codes here, be sure you have all A, A-, ... thru D, and F grades completed.
  # gradepoint = ???
  grade_dict = {'A': 4., 'A-':3.7, 'B+':3.3, 'B':3., 'B-':2.7, \
                'C+':2.3, 'C':2., 'C-':1.7, 'D':1., 'D+':1.3, 'D-':0.7, 'F':0.}
  gradepoint = grade_dict.get(grade, None)
  if not gradepoint:
    raise ValueError("Invalid grade")
  return gradepoint


# Try:
print(to_gradepoint(grade))

# What is the input (function argument) data type for find_grade? 
# What is the output (function return) data type for find_grade(grade) ?

#%% [markdown]
# The input to to_gradepoint is a String value that represent the letter grade, and the function returns the gradepoint, which is a float value 

#%%
###################################### Question 4 ###############################
# next the function to_gradepoint_credit(course)
# which calculates the total weight grade points you earned in one course. Say A- with 3 credits, that's 11.1 total grade_point_credit
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

def to_gradepoint_credit(course):
  """
  Computes gradepoint credit from grade and credits information
  :param course: dictionary that contains all information required to compute gradepoint credit
  :return: Float value representing gradepoint credit
  """
  # write an appropriate and helpful docstring
  # ??????    fill in your codes here
  # grade_point_credit = ?????
  # eventually, if you need to print out the value to 2 decimal, you can 
  # try something like this for floating point values %f
  # print(" %.2f " % grade_point_credit)
  grade_point_credit = to_gradepoint(course["grade"]) * course["credits"]
  return grade_point_credit

# Try:
print(" %.2f " % to_gradepoint_credit(course) )

# What is the input (function argument) data type for to_gradepoint_credit? 
# What is the output (function return) data type for to_gradepoint_credit(course) ?
#%% [markdown]
# to_gradepoint_credits takes in course, which is a dictionary that contains several information about the course and returns a float value that represent the gradepoint credit

#%%
###################################### Question 5 ###############################
# next the function gpa(courses) to calculate the GPA 
# It is acceptable syntax for list, dictionary, JSON and the likes to be spread over multiple lines.
courses = [ 
  { "class":"Intro to DS", "id":"DATS 6101", "semester":"spring", "year":2020, "grade":'B-', "credits":3 } , 
  { "class":"Data Warehousing", "id":"DATS 6102", "semester":"fall", "year":2020, "grade":'A-', "credits":4 } , 
  { "class":"Intro Data Mining", "id":"DATS 6103", "semester":"spring", "year":2020, "grade":'A', "credits":3 } ,
  { "class":"Machine Learning I", "id":"DATS 6202", "semester":"fall", "year":2020, "grade":'B+', "credits":4 } , 
  { "class":"Machine Learning II", "id":"DATS 6203", "semester":"spring", "year":2021, "grade":'A-', "credits":4 } , 
  { "class":"Visualization", "id":"DATS 6401", "semester":"spring", "year":2021, "grade":'C+', "credits":3 } , 
  { "class":"Capstone", "id":"DATS 6101", "semester":"fall", "year":2021, "grade":'A-', "credits":3 } 
  ]

def __extract_credit(course):
  """
  extract_credit is a helper function that is truly meant for internal purpose
  and it helps in extracting credits value from the course dictionary
  
  :param course: a dictionary that contains several information about the course itself
  :return: an integer value representing the number of credits for the given course
  """

  return course["credits"]

def find_gpa(courses):
  """
  find_gpa computes the gpa from the grades and credits information altogether.
  :params courses: a list of dictionaries each consisting of course information
  :return: a float value representing the gpa

  """
  # write an appropriate and helpful docstring
  total_grade_point_credit = sum(list(map(to_gradepoint_credit, courses)))
  total_credits = sum(list(map(__extract_credit, courses)))
  gpa = total_grade_point_credit/total_credits
  return gpa

# Try:
print(" %.2f " % find_gpa(courses) )

# What is the input (function argument) data type for find_gpa? 
# What is the output (function return) data type for find_gpa(courses) ?

#%% [markdown]
# The input to find_gpa is a list of dictionaries in which each element contains information about the course.
# The function returns a float value representing the gpa.


#%%
###################################### Question 6 ###############################
# Write a function to print out a grade record for a single class. 
# The return statement for such functions should be None or just blank
# while during the function call, it will display the print.
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

def printCourseRecord(course):
  """
  This function does a formatted print of a single course information

  :params course: a dictionary that contains information about the course
  :returns: None
  """
  # write an appropriate and helpful docstring
  # use a single print() statement to print out a line of info as shown here
  # 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
  # ??????    fill in your codes here
  year = course['year']
  cname = course['class']
  cid = course['id']
  sem = course['semester']
  grade = course['grade']
  credits = course['credits']
  gpc = round(to_gradepoint_credit(course), 2)
  print(f'{year} {sem} - {cid} : {cname} ({credits} credits) {grade} Grade point credits: {gpc}')
  return None
  
# Try:
printCourseRecord(course)

# What is the input (function argument) data type for printCourseRecord? 
# What is the output (function return) data type for printCourseRecord(course) ?
#%% [markdown]
# printCourseRecord aims at printing a single course record, and takes in a dictionary course.
# It does not return a value, in other words, it returns None.

#%%
###################################### Question 7 ###############################
# write a function (with arguement of courses) to print out the complete transcript and the gpa at the end
# 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
# 2018 fall - DATS 6102 : Data Warehousing (4 credits) A-  Grade point credits: 14.80 
# ........  few more lines
# Cumulative GPA: ?????
 
def printTranscript(courses):
  """
  This function prints information of multiple courses and eventually the CGPA.

  :param courses: a list of dictionaries in which each element contains information about the course
  :return: None
  """
  for course in courses:
    printCourseRecord(course)
  
  # after the completion of the loop, print out a new line with the gpa info
  gpa = round(find_gpa(courses), 2)
  print(f'Cumulative GPA: {gpa}')

  return None

# Try to run, see if it works as expected to produce the desired result
# courses is already definted in Q4
printTranscript(courses)
# The transcript should exactly look like this: 
# 2020 spring - DATS 6101 : Intro to DS (3 credits) B- Grade point credits: 8.10
# 2020 fall - DATS 6102 : Data Warehousing (4 credits) A- Grade point credits: 14.80
# 2020 spring - DATS 6103 : Intro Data Mining (3 credits) A Grade point credits: 12.00
# 2020 fall - DATS 6202 : Machine Learning I (4 credits) B+ Grade point credits: 13.20
# 2021 spring - DATS 6203 : Machine Learning II (4 credits) A- Grade point credits: 14.80
# 2021 spring - DATS 6401 : Visualization (3 credits) C+ Grade point credits: 6.90
# 2021 fall - DATS 6101 : Capstone (3 credits) A- Grade point credits: 11.10
# Cumulative GPA: 3.37

# What is the input (function argument) data type for printTranscript? 
# What is the output (function return) data type for printTranscript(courses) ?

#%% [markdown]
# printTranscript takes in courses, which is a list of dictionaries each containing information about course.
# This function does not return a value, in other words, it returns None


#%% 
# ######  QUESTION 8   Recursive function   ##########
# Write a recursive function that calculates the Fibonancci sequence.
# The recusive relation is fib(n) = fib(n-1) + fib(n-2), 
# and the typically choice of seed values are fib(0) = 0, fib(1) = 1. 
# From there, we can build fib(2) and onwards to be 
# fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8, fib(7)=13, ...
# Let's set it up from here:

def fib(n):
  """
  Finding the Fibonacci sequence with seeds of 0 and 1
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is fib(n) = fib(n-1) + fib(n-2)
  :param n: the index, starting from 0
  :return: the sequence
  """
  if n>1:
    return fib(n-1) + fib(n-2)
  elif n == 0:
    return 0
  else:
    return 1

# Try:
for i in range(12):
  print(fib(i))  



#%% 
# ######  QUESTION 9   Recursive function   ##########
# Similar to the Fibonancci sequence, let us create one (say dm_fibonancci) that has a  
# modified recusive relation dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3). 
# Pay attention to the coefficients and their signs. 
# And let us choose the seed values to be dm_fibonancci(0) = 1, dm_fibonancci(1) = 1, dm_fibonancci(2) = 2. 
# From there, we can build dm_fibonancci(3) and onwards to be 1,1,2,3,6,10,...
# Let's set it up from here:

def dm_fibonancci(n):
  """
  Finding the dm_Fibonacci sequence with seeds of 1, 1, 2 for n = 0, 1, 2 respectively
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3)
  :param n: the index, starting from 0
  :return: the sequence
  """
  # assume n is positive integer
  # ??????    fill in your codes here
  if n >2:
    return dm_fibonancci(n-1) + 2*dm_fibonancci(n-2) - dm_fibonancci(n-3)
  elif n == 2:
    return 2
  else:
    return 1

  #return # return what ????

for i in range(12):
  print(dm_fibonancci(i))  # should gives 1,1,2,3,6,10,...


#%%

