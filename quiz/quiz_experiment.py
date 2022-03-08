#%%
import q1data as q1
# After the import, you will have a dictionary called 
q1.courselist
print(f"Length: {len(q1.courselist)}")
# For convenience, let us give it a local name
courses = q1.courselist 
print(courses[0])

#%%

total_time = 0
for course in courses:
    total_time += course['time']
print(f'total_time: {total_time}')


# %%
