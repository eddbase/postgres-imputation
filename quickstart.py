import psycopg2
from psycopg2 import extras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB

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


conn = connect(param_dic)
data = load_iris(as_frame=True, return_X_y=True)

df_train, df_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.33, random_state=42)
df_train["target"] = y_train
df_test["target"] = y_test

df_train = df_train.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})
df_test = df_test.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})

df_train = df_train.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
df_test = df_test.reset_index(drop=True).reset_index().rename(columns={'index':'id'})

cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS iris_train;')
cur.execute('DROP TABLE IF EXISTS iris_test;')
cur.close()
cur = conn.cursor()
cur.execute('create table iris_train (id integer primary key, s_length float, s_width float, p_length float, p_width float, target integer);')
cur.execute('create table iris_test (id integer primary key, s_length float, s_width float, p_length float, p_width float, target integer);')
conn.commit()
cur.close()
execute_batch(conn, df_train, "iris_train")
execute_batch(conn, df_test, "iris_test")

#train linear regression
df_train_encoded = pd.get_dummies(df_train, columns=['target'])
df_test_encoded = pd.get_dummies(df_test, columns=['target'])
reg = LinearRegression().fit(df_train_encoded.drop(["s_length", "id"], axis=1), df_train_encoded["s_length"])
print("SKLearn R2: ", reg.score(df_test_encoded.drop(["s_length", "id"], axis=1), df_test_encoded["s_length"]))

#use our lib
#train
cur = conn.cursor()
cur.execute("SELECT linregr_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['target'], 'iris_train', 1, 0.001, 0, 1000000, false, true);")
rows = cur.fetchall()
conn.commit()
row = rows[0]
cur.close()
#print("params: ", row)
#predict
cur = conn.cursor()
cur.execute("SELECT linregr_predict(ARRAY"+str(row[0])+", ARRAY[ s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[], false, true) as prediction, id from iris_test;")
rows_pred = cur.fetchall()
conn.commit()
#score
ids = []
preds = []
for row in rows_pred:
    ids += [row[1]]
    preds+=[row[0]]
cur.close()

d = {'id': ids, 'pred': preds}
df = pd.DataFrame(data=d)
df = df_test_encoded[["id", "s_length"]].merge(df, left_on='id', right_on='id')

print("Postgres R2: ", r2_score(df["s_length"], df["pred"]))

##### LDA now:

#SKLearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage=0)
clf.fit(df_train.drop(["target", "id"], axis=1), df_train["target"])
print("Accuracy SKLearn LDA: ", clf.score(df_test.drop(["target", "id"], axis=1), df_test["target"]))

#PostgreSQL
cur = conn.cursor()
cur.execute("SELECT lda_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['target'], 'iris_train', 1, 0, false);")
rows = cur.fetchall()
conn.commit()
row = rows[0]
cur.close()
#print("params: ", row)

cur = conn.cursor()
cur.execute("SELECT lda_predict(ARRAY"+str(row[0])+"::float8[], ARRAY[s_length, s_width, p_length, p_width], ARRAY[]::int[], false), id from iris_test;")
rows_pred = cur.fetchall()
conn.commit()

ids = []
preds = []
for row in rows_pred:
    ids += [row[1]]
    preds+=[row[0]]
cur.close()

d = {'id': ids, 'pred': preds}
df = pd.DataFrame(data=d)
df = df_test[["id", "target"]].merge(df, left_on='id', right_on='id')

from sklearn.metrics import accuracy_score
print("Accuracy PostgreSQL LDA: ", accuracy_score(df["target"], df["pred"]))


#QDA now

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis(store_covariance=True)
#df_train_std["target"] = df_train["target"]
clf.fit(df_train.drop(["target", "id"], axis=1), df_train["target"])
print("Accuracy QDA SKLearn ", clf.score(df_test.drop(["target", "id"], axis=1), df_test["target"]))


cur = conn.cursor()
cur.execute("SELECT qda_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['target'], 'iris_train', 1, false);")
rows = cur.fetchall()
conn.commit()
row = rows[0]
cur.close()
#print("params: ", row)

cur = conn.cursor()
cur.execute("SELECT qda_predict(ARRAY"+str(row[0])+"::float8[], ARRAY[s_length, s_width, p_length, p_width], ARRAY[]::int[], false), id from iris_test;")
rows_pred = cur.fetchall()
conn.commit()

ids = []
preds = []
for row in rows_pred:
    ids += [row[1]]
    preds+=[row[0]]
cur.close()

d = {'id': ids, 'pred': preds}
df = pd.DataFrame(data=d)
df = df_test[["id", "target"]].merge(df, left_on='id', right_on='id')

print("Accuracy PostgreSQL QDA: ", accuracy_score(df["target"], df["pred"]))

clf = GaussianNB()
clf.fit(df_train.drop(["target", "id"], axis=1), df_train["target"])
print("Accuracy SKLearn Gaussian NB ", clf.score(df_test.drop(["target", "id"], axis=1), df_test["target"]))


cur = conn.cursor()
cur.execute("SELECT nb_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['target'], 'iris_train', 1, false);")
rows = cur.fetchall()
conn.commit()
row = rows[0]
cur.close()
#print("params: ", row)
cur = conn.cursor()
cur.execute("SELECT nb_predict(ARRAY"+str(row[0])+"::float8[], ARRAY[s_length, s_width, p_length, p_width], ARRAY[]::int[]), id from iris_test;")
rows_pred = cur.fetchall()
conn.commit()

ids = []
preds = []
for row in rows_pred:
    ids += [row[1]]
    preds+=[row[0]]
cur.close()

d = {'id': ids, 'pred': preds}
df = pd.DataFrame(data=d)
df = df_test[["id", "target"]].merge(df, left_on='id', right_on='id')

print("Accuracy PostgreSQL NB: ", accuracy_score(df["target"], df["pred"]))
