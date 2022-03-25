#%%
# based on example in 16-Pandas_Missing_Data.py
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
df = pd.DataFrame(np.random.randn(15, 3),index = pd.date_range('1/1/2021', periods=15), columns = ['A', 'B', 'C'])
print(df)

#%%
df13 = pd.DataFrame(np.random.randn(5, 3), index=['h', 'c', 'a', 'f','e'],columns=['one', 'two', 'three'])
print(df13)
df14 = df13.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(df14)
# remember that index is immutable. You cannot change them, although you can re-order them.

#%%
print(df14['one'].isnull())
print(df14['one'].notnull())
#%%
print(df14['one'].sum())
print(df14['two'].sum())
print(df14.fillna(0)) # remember none of these functions/methods change the df itself.
print(df14)

#%%
df15 = df13.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(df15)
print('Dropping na\n', df15.dropna())
# any row with any na will be dropped

#%%
df16 = df13.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
df16.loc['b','two']=3.14 ; df16.loc['d','two']=0 ; df16.loc['g','two']=-1  
print(df16)
print('Dropping na\n', df16.dropna())
# df16 = df16.dropna()
print(df16)
# any row without any na will be kept

#%%
print(df15)
print('Dropping df15 na on axis 1\n', df15.dropna(axis=1))
# any column with any na will be dropped
#%%
print(df16)
print('Dropping df16 na on axis 1\n', df16.dropna(axis=1))
# columns without any na will be kept

#%%
# Simple imputation using Pandas.replace( )
#
print(df15.head(), '\n')
print(df15.replace({np.NaN:0,df15.loc['a','three']:np.NaN}) , '\n')
print("unchanged\n", df15.head(), '\n')

print('#',50*"-")
#%%
# Categorical encoding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
import dm6103 as dm
print("\nReady to continue.")


#%%
# load the dataframe
dfgap = dm.api_dsLand('gapminder','id')
print("\nReady to continue.")


#%%
# dfgap.head()
dfContCode = pd.get_dummies(dfgap.continent, prefix="cont")
print(dfContCode)

#%%
# To add/join these to the original df, we can do this
dfgap = pd.merge(dfgap, dfContCode, on='id') 
print(dfgap)
# Also see sample codes from 18-Merging_and_Joining.py for left/right join, etc.

#%% [markdown] 
# # Categorical encoding 
# In general for machine learning, the system is not very good at handling categorical 
# variables. By converting them into numerics. 
# Care should be given if the variable is "ordinal", the encoding should preserve the order.
# 
# We will also use other functions in scikitlearn to perform OneHotEncoder, and 
# LabelBinarizer to perform these encodings. 


# %%
