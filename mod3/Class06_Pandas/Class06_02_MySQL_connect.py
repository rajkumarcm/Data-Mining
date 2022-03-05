#%%
# %pip3 install mysql-connector-python
# If the above line does not work, try the version below
# %pip install mysql-connector-python==8.0.28
# Also see https://dev.mysql.com/downloads/connector/python/ for more info and other option


#%%
# See https://pynative.com/python-mysql-database-connection/
#
def dbCon_dsLand(tbname, ind_col_name = ""):
  """ 
  connect to datasci.land database for this class and pull all rows from table
  :param str tbname: table name that exist on the server 
  :param str ind_col_name: optional, name of index column 
  :return: pandas.Dataframe
  """
  
  import mysql.connector
  from mysql.connector import Error
  import pandas as pd

  df = None # set a global variable to store the dataframe
  hostname = 'pysql.datasci.land'
  dbname = 'datascilanddb0'
  username = '6103_sp22'
  pwd = 'v8rX91jb7s'
  query = 'SELECT * FROM `'+ dbname +'`.`'+ tbname + '`'

  try:
    connection = mysql.connector.connect(host=hostname, database=dbname, user=username, password=pwd)
    if connection.is_connected():
      # optional output
      db_Info = connection.get_server_info()
      print(f'Connected to MySQL Server version {db_Info}')
      cursor = connection.cursor()
      cursor.execute("select database();")
      record = cursor.fetchone()
      print(f"You're connected to database: {record}")
      # read query into dataframe df
      df = pd.read_sql(query, connection, index_col= ind_col_name) if (ind_col_name) else pd.read_sql(query, connection) # tables often have unique Id field
      print(f'Dataframe is loaded.')
      cursor.close()
      connection.close()
      print("MySQL connection is closed")

  except Error as e:
    print(f'Error while connecting to MySQL {e}')
      
  return df

# Sample
# d = dbCon_dsLand('AStudentRecord')
# d = dbCon_dsLand('AStudentRecord','id')

print("\nFunction dbCon_dsLand loaded. Ready to continue.")

#%%