
import psycopg2
from psycopg2 import extras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error


param_dic = {
    "host"      : "localhost",
    "database"  : "postgres",
    "user"      : "postgres",
    "password"  : ""
}

def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)
    print("Connection successful")
    return conn

def execute_batch(conn, df, table, page_size=100):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s)" % (table, cols)
    print(query)
    cursor = conn.cursor()
    try:
        extras.execute_batch(cursor, query, tuples, page_size)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_batch() done")
    cursor.close()
    
def nan_to_null(f,
        _NULL=psycopg2.extensions.AsIs('NULL'),
        _Float=psycopg2.extensions.Float):
    if not np.isnan(f):
        return _Float(f)
    return _NULL

psycopg2.extensions.register_adapter(float, nan_to_null)

conn = connect(param_dic)

#ADD NULLS

data = load_iris(as_frame=True, return_X_y=True)
data[0]["target"] = data[1]
data = data[0]
np.random.seed(0)
for col in ["sepal width (cm)", "petal length (cm)", "petal width (cm)"]:
    idxs = np.random.choice(len(data), int(len(data)*0.2), replace=False)
    data.loc[idxs, col] = np.nan


data = data.reset_index(drop=True).reset_index().rename(columns={'index':'id', 'sepal length (cm)':'s_length', 'sepal width (cm)':'s_width','petal length (cm)': 'p_length', 'petal width (cm)': 'p_width'})
#copy table to PostgreSQL
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS iris_impute;')
cur.close()
cur = conn.cursor()
cur.execute('create table iris_impute (id integer primary key, s_length float, s_width float, p_length float, p_width float, target integer);')
conn.commit()
cur.close()
execute_batch(conn, data, "iris_impute")

data_c = data.copy()
#impute with SKLearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(estimator=LinearRegression(), random_state=0, max_iter=2, imputation_order='roman')
data_imputed = imp_mean.fit_transform(data_c)
data_imputed = pd.DataFrame(data_imputed, columns = ['id','s_length', 's_width', 'p_length', 'p_width', 'target'])

#IMPUTE WITH OUR LIB

conn.commit()
conn.autocommit = True

queries = ["CALL MICE_high('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);", "CALL MICE_low('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);","CALL MICE_baseline('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);"]


for q in queries:
    #impute
    print(q)
    cur = conn.cursor()
    cur.execute(q)
    cur.close()
    conn.commit()
    
    ##fetch and compare

    cur = conn.cursor()
    cur.execute("SELECT id, s_width, p_length, p_width from iris_impute_res;")
    rows_pred = cur.fetchall()
    conn.commit()
    #score
    ids = []
    s_width = []
    p_length = []
    p_width = []
    for row in rows_pred:
        ids += [row[0]]
        s_width += [row[1]]
        p_length += [row[2]]
        p_width += [row[3]]
    cur.close()

    d = {'id': ids, 's_width': s_width, 'p_length':p_length , 'p_width':p_width }
    df_postgres = pd.DataFrame(data=d).sort_values(by=['id']).reset_index(drop=True)

    for miss_col in ['s_width', 'p_length', 'p_width']:
        bool_mask = np.isnan(data_c[miss_col])
        print("MSE between SKLearn and PostgreSQL in col: ",miss_col ," : ", mean_squared_error(df_postgres["s_width"][bool_mask], data_imputed["s_width"][bool_mask]))
