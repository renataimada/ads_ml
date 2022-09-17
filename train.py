# Databricks notebook source
# DBTITLE 1,Imports
from sklearn import tree
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics

import mlflow

# COMMAND ----------

# DBTITLE 1,Dados
print("Obtendo dados...")

df = spark.table("sandbox_apoiadores.abt_dota_pre_match").toPandas()
df

# COMMAND ----------

# DBTITLE 1,Setup do experimento
exp_name = "/Users/renatanimada@gmail.com/fatec_dota_renata"
mlflow.set_experiment(exp_name)

# COMMAND ----------

# DBTITLE 1,Definições de coluna
target = "radiant_win"
id_column = "match_id"

features = list(set(df.columns) - set([target, id_column]))

# COMMAND ----------

# DBTITLE 1,Split de dados
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], 
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42)

# COMMAND ----------

print("Geral:", df[target].mean())
print("Treino:", y_train.mean())
print("Teste:", y_test.mean())

# COMMAND ----------

with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    #model = tree.DecisionTreeClassifier(min_samples_leaf=50)
    #model = ensemble.RandomForestClassifier(min_samples_leaf=15)
    model = ensemble.RandomForestClassifier(min_samples_leaf=30)
    #model = ensemble.RandomForestClassifier(n_estimators=50, min_samples_leaf=15)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)

    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Acurácia:", acc_train)

    y_test_pred = model.predict(X_test)

    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acurácia:", acc_test)

# COMMAND ----------

y_train_pred = model.predict(X_train)

acc_train = metrics.accuracy_score(y_train, y_train_pred)
print("Acurácia:", acc_train)

# COMMAND ----------

y_test_pred = model.predict(X_test)

acc_test = metrics.accuracy_score(y_test, y_test_pred)
print("Acurácia:", acc_test)
