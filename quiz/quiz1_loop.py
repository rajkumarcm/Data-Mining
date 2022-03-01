#%%
import q1data as q1
# After the import, you will have a dictionary called 
q1.courselist
print(f"Length: {len(q1.courselist)}")
# For convenience, let us give it a local name
courses = q1.courselist 
print(courses[0])

#%%[markdown]
# Do not import any other libraries to perform this task.
# 
# I have an agressive plan to take all the python courses on a website, within the month of March. 
# To make a plan, I need to loop through all the courses, starting March 1 (2022) which is a 
# Tuesday, and assuming I will spend 8 hours a day, 7 days a week, to finish them all in March.
# The schedule is printed out like this:
# 
# Mar 1 (Tue): Supervised Learning with scikit-learn : 4.0 hour
# Mar 1 (Tue): Python Data Science Toolbox (Part 1) : 3.0 hour
# Mar 1 (Tue): Introduction to Python : 4.0 hour
# Mar 2 (Wed): Intermediate Python : 4.0 hour
# Mar 2 (Wed): Introduction to Data Science in Python : 4.0 hour
# Mar 3 (Thu): Data Manipulation with pandas : 4.0 hour
# Mar 3 (Thu): Python Data Science Toolbox (Part 2) : 4.0 hour
# Mar 4 (Fri): Joining Data with pandas : 4.0 hour
# Mar 4 (Fri): Introduction to Data Visualization with Matplotlib : 4.0 hour
# Mar 5 (Sat): Introduction to Importing Data in Python : 3.0 hour
# Mar 5 (Sat): Cleaning Data in Python : 4.0 hour
# Mar 6 (Sun): Introduction to Data Visualization with Seaborn : 4.0 hour
# Mar 6 (Sun): Statistical Thinking in Python (Part 1) : 3.0 hour
# Mar 7 (Mon): Writing Efficient Python Code : 4.0 hour
# Mar 7 (Mon): Exploratory Data Analysis in Python : 4.0 hour
# ..........
# 
# As you can see, in the first day, I can complete a little more than 2 courses, 
# then start the third course "Introduction to Python". I will resume this third 
# course on day 2 with 3 hours of learning left (without printing out 
# that line again for March 2, but keep track of how many hours left for the 
# day to learn). So continue with the next courses, and stop after 1 hour 
# of "Introduction to Data Science in Python" on March 2, etc.
# 
# Print out the entire schedule for all the courses. It is an 
# agressive 30-day plan.
# 
# 

# %%
dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat')
marchdate = 1     # March dates from 1, 2, ... , 31, increment by 1 after each filled day
hourofdayleft = 8   # Keep a running total of hours left after taken a course. Starting with 8 hours on the first day
# loop through the list of courses and print.

# %%
def get_n_courses(start_idx, n_courses, hours_carried, day):
    n = 0
    hours_can_do = hourofdayleft - hours_carried
    for i in range(start_idx, n_courses):
        hours_can_do -= courses[i]['time']
        n += 1
        if day == 1 and n == 3:
            break
        if hours_can_do <= 0:
            break
    return n, abs(hours_can_do) # the returning hours_can_do is hours carried over


#%%
left_over = None
n_courses = len(courses)
course_idx = 0
hrs_carried = 0
tmp_t = 0
for i in range(3):
    tmp_t += courses[i]['time'] 

for d in range(1, 31): # since you said 30-day plan
    didx = ((d+1)%7)
    dayofweek = dayofweektuple[didx]
    n_courses_to_do, hrs_carried = get_n_courses(course_idx, n_courses, hrs_carried, d)
    for i in range(course_idx, course_idx + n_courses_to_do):
        course = courses[i]
        print(f"Mar {d} ({dayofweek}): {course['title']} : {course['time']} hour")
    course_idx = course_idx + n_courses_to_do
    # if d == 1:
    #     course = courses[course_idx]
    #     course_idx += 1
    #     hrs_carried = hourofdayleft - tmp_t
    #     print(f"Mar {d} ({dayofweek}) : {course['title']} : {course['time']} hour")


# %%
