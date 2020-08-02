#!/usr/bin/env python
# coding: utf-8

# # **Imports and Constants**

# In[11]:


# find spark
#import findspark
# findspark.init()


# In[14]:


# imports
#from __future__ import print_function
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import PCA
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql.functions import col
from functools import reduce
from pyspark import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, size, isnan, array_contains, when, count, pandas_udf, PandasUDFType
from pyspark.sql.types import *
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import pathlib
import json
import datetime
import numpy as np
import time
import os
import random
import statistics
import pathlib
import platform

# comment for converting to .py file
########################################################
# env variables
# if platform.system() == 'Windows':
#     %env PYSPARK_DRIVER_PYTHON = python
#     %env PYSPARK_PYTHON = python
# elif platform.system() == 'Linux':
#     %env PYSPARK_DRIVER_PYTHON = python
#     %env PYSPARK_PYTHON = python3
# else:
#     %env PYSPARK_DRIVER_PYTHON = python3.6
#     %env PYSPARK_PYTHON = python3.6

# incompatibility with Pyarrow
# need to install Pyarrow 0.14.1 or lower or Set the environment variable ARROW_PRE_0_15_IPC_FORMAT=1
# %env ARROW_PRE_0_15_IPC_FORMAT = 1
########################################################

# used versions:
# spark='2.4.3' python='3.6' pyarrow='0.14.1'

# for new system:
# import findspark
# findspark.init()
# %pip install numpy
# %pip install -U matplotlib
# %pip install pandas
# %pip install Pyarrow==0.14.0
# %env PYSPARK_DRIVER_PYTHON=python
# %env PYSPARK_PYTHON=python


# In[15]:


# paths

BASE_PATH = pathlib.Path().absolute()

KMEANS_REL_PATH = os.path.join(os.path.join(os.path.join(
    "Datasets", "Smart*"), "apartment"), "kmeans models")
DATASET_REL_PATH = os.path.join(
    os.path.join("Datasets", "Smart*"), "apartment")


DATASET_PATH = os.path.join(BASE_PATH, DATASET_REL_PATH)
KMEANS_PATH = os.path.join(BASE_PATH, KMEANS_REL_PATH)

print(DATASET_PATH)
print(KMEANS_PATH)

#from google.colab import drive
# drive.mount('/gdrive')


# # **Dataset**

# In[16]:


# load and save .read_pickle() and .to_pickle()

# save
# dataset.to_pickle(DATASET_PATH+"dataset.pkl")
# aggregated_dataset.to_pickle(DATASET_PATH+"aggregated_dataset.pkl")
# json_dataset.to_pickle(DATASET_PATH+"json_dataset.pkl")
# dataset.to_csv(DATASET_PATH+"dataset.csv")
# aggregated_dataset.to_csv(DATASET_PATH+"aggregated_dataset.csv")
# json_dataset.to_csv(DATASET_PATH+"json_dataset.csv")
# aggregated_dataset_rowBased.to_csv(DATASET_PATH+"aggregated_dataset_rowBased.csv")

# load
#dataset = pd.read_pickle(os.path.join(DATASET_PATH, 'dataset.pkl'))
#aggregated_dataset = pd.read_pickle(os.path.join(DATASET_PATH, 'aggregated_dataset.pkl'))


# # **Functions**

# ## **Malicious Samples**

# In[17]:


# Generate malicious samples
def h1(x):
    MAX = 0.8
    MIN = 0.1
    alpha = random.uniform(MIN, MAX)
    temp = np.array(x)
    return (temp*alpha).tolist()


def h2(x):
    MIN_OFF = 4  # hour
    DURATION = random.randint(MIN_OFF, 23)
    START = random.randint(0, 23-DURATION) if DURATION != 23 else 0
    END = START+DURATION
    temp = []
    for i in range(len(x)):
        if i < START or i >= END:
            temp.append(x[i])
        else:
            temp.append(0.0)
    return temp


def h3(x):
    MAX = 0.8
    MIN = 0.1
    temp = []
    for i in range(len(x)):
        temp.append(x[i]*random.uniform(MIN, MAX))
    return temp


def h4(x):
    MAX = 0.8
    MIN = 0.1
    mean = np.mean(x)
    temp = []
    for i in range(len(x)):
        temp.append(mean*random.uniform(MIN, MAX))
    return temp


def h5(x):
    MAX = 0.8
    MIN = 0.1
    mean = np.mean(x)
    temp = []
    for i in range(len(x)):
        temp.append(mean)
    return temp


def h6(x):
    temp = np.array(x)
    # temp=temp[::-1]
    temp = np.flipud(temp)
    return temp.tolist()


# add malicious samples
def create_malicious_df(sdf):
    def random_attack_assigner(x):
        NUMBER_OF_MALICIOUS_GENERATOR = 6
        res = []
        for row in x:
            rand = random.randint(1, NUMBER_OF_MALICIOUS_GENERATOR)
            if rand == 1:
                temp = (h1(row))
            elif rand == 2:
                temp = (h2(row))
            elif rand == 3:
                temp = (h3(row))
            elif rand == 4:
                temp = (h4(row))
            elif rand == 5:
                temp = (h5(row))
            elif rand == 6:
                temp = (h6(row))
            res.append(temp)
        return pd.Series(res)
    random_attack_assigner_UDF = pandas_udf(
        random_attack_assigner, returnType=ArrayType(FloatType()))
    # sdf_malicious=sdf
    N = False
    sdf = sdf.withColumn("N", f.lit(N))  # malicious sample
    # change '#' column number to negative
    sdf = sdf.withColumn("#", col("#")*-1)
    sdf = sdf.withColumn("power", random_attack_assigner_UDF(col("power")))
    # sdf=sdf.drop('statistics')
    sdf = sdf.withColumn("statistics", generate_feature_UDF(col("power")))
    sdf = add_statistics_column(sdf)  # for update statistics
    return sdf.select(sdf.columns)  # to reorder columns


# In[19]:


# plot
def plot_malicious_samples(presence=[True, True, True, True, True, True, True]):

    read_value = [3.4803431034088135, 2.529871702194214, 2.2175486087799072, 2.629481077194214, 2.9629790782928467, 2.0697860717773438, 2.900712251663208, 2.926414966583252, 4.8191237449646, 4.156486988067627, 2.6474769115448, 2.1933677196502686,
                  2.261159658432007, 2.340345621109009, 2.7386586666107178, 3.2414891719818115, 1.8946533203125, 3.1397650241851807, 2.8951449394226074, 3.4589333534240723, 2.726524829864502, 6.511429309844971, 3.4918391704559326, 3.787257432937622]
    lists = []
    colors = ['b', 'r-', 'g--', 'c:', 'm-.', 'y-', 'k--']
    if presence[0] == True:
        lists.append(read_value)
    if presence[1] == True:
        lists.append(h1(read_value))
    if presence[2] == True:
        lists.append(h2(read_value))
    if presence[3] == True:
        lists.append(h3(read_value))
    if presence[4] == True:
        lists.append(h4(read_value))
    if presence[5] == True:
        lists.append(h5(read_value))
    if presence[6] == True:
        lists.append(h6(read_value))
    #font = {'size': 12}
    #plt.rc('font', **font)
    plt.figure(num=None, figsize=(28, 16), dpi=120,
               facecolor='w', edgecolor='k')
    plt.xlabel("time (hour)", fontsize=25)
    plt.ylabel("usage (kw)", fontsize=25)
    #plt.title("malicious samples")
    plt.xticks(np.arange(0, 24, step=1))
    plt.plot(read_value)
    for i in range(len(lists)):
        if i == 0:
            plt.plot(lists[i], colors[i], label='normal usage')
        else:
            plt.plot(lists[i], colors[i], label='attack %s' % i)
    # plt.legend()
    plt.legend(prop={'size': 20})
    plt.savefig('attack.pdf', bbox_inches='tight')
    plt.savefig('attack.png', bbox_inches='tight')
    # plt.savefig('attack.eps', format='eps')
    # plt.show()
    return None

# plot_malicious_samples([True,True,True,True,True,True,True])


# ## **Prepare Spark Dataset**

# In[8]:


# rename columns
def rename_dataframe(sdf):
    names = ['#', 'date', 'id', 'power']
    for c, n in zip(sdf.columns, names):
        sdf = sdf.withColumnRenamed(c, n)
    return sdf

# sdf=rename_dataframe(sdf)
# sdf.show()

# convert power to array


def string_power_to_array(sdf):
    temp = sdf.withColumn("power", f.regexp_replace(f.regexp_replace(f.col("power"), "\\[", ""), "\\]", "")
                          .alias("power"))
    temp = temp.withColumn("power", split(col("power"), ",\s*")
                           .cast(ArrayType(FloatType())).alias("power"))
    return temp

# sdf=string_power_to_array(sdf)
# sdf.show()


def add_validation_column(sdf):
    def validation(x):
        res = []
        for row in x:
            v = True
            if (len(row) != 24 or  # unusual size
                (row >= 0).sum() != 24 or  # number of valid elements = 24
                # sum(n >= 0 for n in row) != 24 or
                # equal or more than 3 zero elements
                np.count_nonzero(row == 0) >= 3 or
                    sum(n < 0 for n in row) > 0):  # not have negative element
                v = False
            res.append(v)
        return pd.Series(res)
    validation_UDF = pandas_udf(validation, returnType=BooleanType())
    temp = sdf.withColumn("V", validation_UDF(col("power")))
    return temp

# sdf=add_validation_column(sdf)
# sdf.show()


# add "N"ormal consumption ("N"onmalicious) column
def add_Normal_column(sdf):
    N = True
    temp = sdf.withColumn("N", f.lit(N))
    return temp

# sdf=add_Normal_column(sdf)
# sdf.show()

# filter data


def filter_dataset(sdf, from_date="BEGIN", to_date="END", ID="*", V="*"):
    temp = sdf
    if (from_date != "BEGIN"):
        temp = temp.filter(sdf.date > from_date)  # filter date (from X)
    if (to_date != "END"):
        temp = temp.filter(sdf.date < to_date)  # filter date (to Y)
    if (ID != "*"):
        temp = temp.filter(sdf.id == ID)  # filter IDs
    if (V != "*"):
        temp = temp.filter(sdf.V == V)  # filter validation
    return temp

# sdf=filter_dataset(sdf,from_date="BEGIN",to_date="END",ID="Apt36",V="True")
# sdf.show()


def split_power(sdf):
    temp = sdf.select("#", "date", "id",
                      sdf.power[0].alias("H0"), sdf.power[1].alias(
                          "H1"), sdf.power[2].alias("H2"), sdf.power[3].alias("H3"),
                      sdf.power[4].alias("H4"), sdf.power[5].alias(
                          "H5"), sdf.power[6].alias("H6"), sdf.power[7].alias("H7"),
                      sdf.power[8].alias("H8"), sdf.power[9].alias(
                          "H9"), sdf.power[10].alias("H10"), sdf.power[11].alias("H11"),
                      sdf.power[12].alias("H12"), sdf.power[13].alias(
                          "H13"), sdf.power[14].alias("H14"), sdf.power[15].alias("H15"),
                      sdf.power[16].alias("H16"), sdf.power[17].alias(
                          "H17"), sdf.power[18].alias("H18"), sdf.power[19].alias("H19"),
                      sdf.power[20].alias("H20"), sdf.power[21].alias("H21"), sdf.power[22].alias("H22"), sdf.power[23].alias("H23"))
    return temp

# split_sdf=split_power(sdf)
# sdf.show()

# make id list


def get_ids(sdf):
    IDs = np.array(sdf.select("id").distinct().collect())
    #IDs = IDs.reshape(1, len(IDs))
    IDs = IDs.flatten()
    return IDs

# number of instances per id


def get_instance_number_per_id(sdf):
    IDs = np.array(sdf.select("id").distinct().collect())
    IDs = IDs.flatten()
    id_instance_dict = {}

    for i in IDs:
        temp = sdf.filter(sdf.id == int(i))
        row_number = temp.count()
        id_instance_dict[int(i)] = row_number


# generate uniqe id
def generate_uniqe_id(sdf):
    temp = sdf
    temp = temp.withColumn("uid", f.concat(
        col("id"), f.lit("-"), col("#")).alias("uid"))
    return temp

# sdf=generate_uniqe_id(sdf)


# generate feature
def generate_feature(x):
    res = []
    for row in x:
        row = np.array(row)  # to numpy
        statistics = []
        min_val = np.nanmin(row)
        max_val = np.nanmax(row)
        mean_val = np.nanmean(row)
        std_val = np.nanstd(row)
        statistics.append(mean_val)
        statistics.append(std_val)
        statistics.append(min_val)
        statistics.append(max_val)
        res.append(statistics)
    return pd.Series(res)


generate_feature_UDF = pandas_udf(
    generate_feature, returnType=ArrayType(FloatType()))


def add_statistics_column(sdf):
    temp = sdf.withColumn("statistics", generate_feature_UDF(col("power")))
    return temp

# sdf=add_statistics_column(sdf)
# sdf.show()


# sdf.show()
# print("number of rows: " + str(sdf.count()))
# sdf.collect()
# sdf.printSchema()
# split_sdf=add_validation_column(split_sdf)


# In[ ]:


# In[ ]:


# # **Models**

# ## **K-Means**

# In[9]:


def prepare_for_kmeans(sdf):

    temp = sdf

    # define function for split power column
    def split(sdf):
        temp = sdf.select("#", "V", "N", "date", "id", "uid",
                          sdf.power[0].alias("H0"), sdf.power[1].alias(
                              "H1"), sdf.power[2].alias("H2"), sdf.power[3].alias("H3"),
                          sdf.power[4].alias("H4"), sdf.power[5].alias(
                              "H5"), sdf.power[6].alias("H6"), sdf.power[7].alias("H7"),
                          sdf.power[8].alias("H8"), sdf.power[9].alias(
                              "H9"), sdf.power[10].alias("H10"), sdf.power[11].alias("H11"),
                          sdf.power[12].alias("H12"), sdf.power[13].alias(
                              "H13"), sdf.power[14].alias("H14"), sdf.power[15].alias("H15"),
                          sdf.power[16].alias("H16"), sdf.power[17].alias(
                              "H17"), sdf.power[18].alias("H18"), sdf.power[19].alias("H19"),
                          sdf.power[20].alias("H20"), sdf.power[21].alias(
                              "H21"), sdf.power[22].alias("H22"), sdf.power[23].alias("H23"),
                          sdf.statistics[0].alias("S0"), sdf.statistics[1].alias("S1"), sdf.statistics[2].alias("S2"), sdf.statistics[3].alias("S3"))
        return temp

    # call the split_power function
    temp = split(temp)

    # filter date
    # temp=temp.filter(temp.date > "2014-08-15").filter(temp.date < "2014-08-19") #filter dates
    # temp=temp.filter(temp.id == "Apt40") #filter IDs
    temp = temp.filter(temp.V == True)  # filter valid rows

    FEATURES = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11',
                'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'S0', 'S1', 'S2', 'S3']

    # call the generate_uniqe_id function
    temp = generate_uniqe_id(temp)

    # make ready
    vecAssembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    #df_kmeans = vecAssembler.transform(temp).select(col("uid").alias("id"), col("features"))
    df_kmeans = vecAssembler.transform(
        temp).select(col("uid"), col("features"))
    return df_kmeans


# In[10]:


# run k-means


def kmeans(sdf_kmeans):
    # find best k
    MAX_k = 8
    costs = np.zeros(MAX_k)
    silhouettes = np.zeros(MAX_k)
    silhouettes[1] = 0  # set value for k=1
    for k in range(2, MAX_k):
        kmeans = KMeans().setK(k).setSeed(1)
        model = kmeans.fit(sdf_kmeans)
        costs[k] = model.computeCost(sdf_kmeans)  # requires Spark 2.0 or later
        predictions = model.transform(sdf_kmeans)
        evaluator = ClusteringEvaluator()
        silhouettes[k] = evaluator.evaluate(predictions)

    # show silhouette
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, MAX_k), silhouettes[2:MAX_k])
    ax.set_xlabel('k')
    ax.set_ylabel('silhouette')

    # show cost
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, MAX_k), costs[2:MAX_k])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')

    # find best k
    best_k = np.argmax(silhouettes)
    print("maximum value of silhouette is: " +
          str(silhouettes[best_k]) + " in index: " + str(best_k))

    # Trains a k-means model.
    kmeans = KMeans().setK(best_k).setSeed(1)
    model = kmeans.fit(sdf_kmeans)

    # Make predictions
    predictions = model.transform(sdf_kmeans)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
#     print("Silhouette with squared euclidean distance = " + str(silhouette))

#     # Shows the result.
#     centers = model.clusterCenters()
#     print("Cluster Centers: ")
#     for center in centers:
#         print(center)

#     transformed = model.transform(sdf_kmeans).select('id', 'prediction')
#     transformed.show()
#     transformed.groupby('prediction').count().show()
#     rows = transformed.collect()
#     prediction = spark.createDataFrame(rows)
#     prediction.show()
    return model, best_k, silhouette  # silhouettes: new


# ## **Decision Tree Methods**

# In[11]:


def prepare_for_decision_tree_methods(sdf):

    temp = sdf

    # define function for split power column
    def split(sdf):
        temp = sdf.select("#", "V", "N", "date", "id", "uid",
                          sdf.power[0].alias("H0"), sdf.power[1].alias(
                              "H1"), sdf.power[2].alias("H2"), sdf.power[3].alias("H3"),
                          sdf.power[4].alias("H4"), sdf.power[5].alias(
                              "H5"), sdf.power[6].alias("H6"), sdf.power[7].alias("H7"),
                          sdf.power[8].alias("H8"), sdf.power[9].alias(
                              "H9"), sdf.power[10].alias("H10"), sdf.power[11].alias("H11"),
                          sdf.power[12].alias("H12"), sdf.power[13].alias(
                              "H13"), sdf.power[14].alias("H14"), sdf.power[15].alias("H15"),
                          sdf.power[16].alias("H16"), sdf.power[17].alias(
                              "H17"), sdf.power[18].alias("H18"), sdf.power[19].alias("H19"),
                          sdf.power[20].alias("H20"), sdf.power[21].alias(
                              "H21"), sdf.power[22].alias("H22"), sdf.power[23].alias("H23"),
                          sdf.statistics[0].alias("S0"), sdf.statistics[1].alias("S1"), sdf.statistics[2].alias("S2"), sdf.statistics[3].alias("S3"))
        return temp

    # call the split_power function
    temp = split(temp)

    # boolean to string (for "N" column)
    temp = temp.withColumn("N", f.col("N").cast('string'))
    # temp.printSchema()

    # filter date
    # temp=temp.filter(temp.date > "2014-08-15").filter(temp.date < "2014-08-19") #filter dates
    # temp=temp.filter(temp.id == "Apt40") #filter IDs
    temp = temp.filter(temp.V == True)  # filter valid rows

    FEATURES = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11',
                'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'S0', 'S1', 'S2', 'S3']

    # call the generate_uniqe_id function
    # temp=generate_uniqe_id(temp)

    # make features ready
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    output = assembler.transform(temp)
    #df_kmeans = vecAssembler.transform(temp).select(col("uid").alias("id"), col("features"))

    # make label ready
    indexer = StringIndexer(inputCol="N", outputCol="NIndex")
    output_fixed = indexer.fit(output).transform(output)

    final_data = output_fixed.select("features", 'NIndex')
    return final_data


# In[12]:


# run decision tree methods
def decision_tree(train_data, test_data):
    dtc = DecisionTreeClassifier(labelCol='NIndex', featuresCol='features')
    rfc = RandomForestClassifier(
        labelCol='NIndex', featuresCol='features')  # ,numTrees=100
    gbt = GBTClassifier(labelCol='NIndex', featuresCol='features')

    dtc_model = dtc.fit(train_data)
    rfc_model = rfc.fit(train_data)
    gbt_model = gbt.fit(train_data)

    dtc_predictions = dtc_model.transform(test_data)
    rfc_predictions = rfc_model.transform(test_data)
    gbt_predictions = gbt_model.transform(test_data)

    # evaluation
    # Select (prediction, true label) and compute test error
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="NIndex", predictionCol="prediction", metricName="accuracy")

    dtc_acc = acc_evaluator.evaluate(dtc_predictions)
    rfc_acc = acc_evaluator.evaluate(rfc_predictions)
    gbt_acc = acc_evaluator.evaluate(gbt_predictions)

    # new:
    # Let's use the run-of-the-mill evaluator
    evaluator = BinaryClassificationEvaluator(labelCol='NIndex')
    # We have only two choices: area under ROC and PR curves :-(
    dtc_auroc = evaluator.evaluate(
        dtc_predictions, {evaluator.metricName: "areaUnderROC"})
    dtc_auprc = evaluator.evaluate(
        dtc_predictions, {evaluator.metricName: "areaUnderPR"})
    rfc_auroc = evaluator.evaluate(
        rfc_predictions, {evaluator.metricName: "areaUnderROC"})
    rfc_auprc = evaluator.evaluate(
        rfc_predictions, {evaluator.metricName: "areaUnderPR"})
    gbt_auroc = evaluator.evaluate(
        gbt_predictions, {evaluator.metricName: "areaUnderROC"})
    gbt_auprc = evaluator.evaluate(
        gbt_predictions, {evaluator.metricName: "areaUnderPR"})
    #print("DT Area under ROC Curve: {:.4f}".format(dtc_auroc))
    #print("DT Area under PR Curve: {:.4f}".format(dtc_auprc))
    #print("RF Area under ROC Curve: {:.4f}".format(rfc_auroc))
    #print("RF Area under PR Curve: {:.4f}".format(rfc_auprc))
    #print("GB Area under ROC Curve: {:.4f}".format(gbt_auroc))
    #print("GB Area under PR Curve: {:.4f}".format(gbt_auprc))

    #print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))
    #print('A random forest ensemble had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))
    #print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc*100))

    return dtc_acc, dtc_auroc, dtc_auprc, rfc_acc, rfc_auroc, rfc_auprc, gbt_acc, gbt_auroc, gbt_auprc


# ## **PCA**

# In[13]:


def prepare_for_pca(sdf):

    temp = sdf

    # define function for split power column
    def split_power(sdf):
        temp = sdf.select("#", "V", "N", "date", "id", "uid",
                          sdf.power[0].alias("H0"), sdf.power[1].alias(
                              "H1"), sdf.power[2].alias("H2"), sdf.power[3].alias("H3"),
                          sdf.power[4].alias("H4"), sdf.power[5].alias(
                              "H5"), sdf.power[6].alias("H6"), sdf.power[7].alias("H7"),
                          sdf.power[8].alias("H8"), sdf.power[9].alias(
                              "H9"), sdf.power[10].alias("H10"), sdf.power[11].alias("H11"),
                          sdf.power[12].alias("H12"), sdf.power[13].alias(
                              "H13"), sdf.power[14].alias("H14"), sdf.power[15].alias("H15"),
                          sdf.power[16].alias("H16"), sdf.power[17].alias(
                              "H17"), sdf.power[18].alias("H18"), sdf.power[19].alias("H19"),
                          sdf.power[20].alias("H20"), sdf.power[21].alias("H21"), sdf.power[22].alias("H22"), sdf.power[23].alias("H23"))
        return temp

    # call the split_power function
    temp = split_power(temp)

    temp = temp.filter(temp.V == True)  # filter valid rows

    FEATURES = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11',
                'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23']

    # call the generate_uniqe_id function
    temp = generate_uniqe_id(temp)

    # make ready
    vecAssembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    #df_kmeans = vecAssembler.transform(temp).select(col("uid").alias("id"), col("features"))
    df_pca = vecAssembler.transform(temp).select(
        "#", "V", "N", "date", "id", "uid", col("features"))
    return df_pca


def pca_for_tree(sdf):
    #sdf = prepare_for_pca(sdf)
    pca = PCA(k=10, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(sdf)
    result = model.transform(sdf).select(
        "NIndex", col("pcaFeatures").alias("features"))
    return result


def pca_for_kmeans(sdf):
    #sdf = prepare_for_pca(sdf)
    pca = PCA(k=10, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(sdf)
    result = model.transform(sdf).select(
        "uid", col("pcaFeatures").alias("features"))
    return result


# sdf_pca=pca(sdf)
# sdf_pca.show() #truncate=False


# ## **MLP**

# In[14]:


def prepare_for_mlp(sdf):

    temp = sdf

    # define function for split power column
    def split(sdf):
        temp = sdf.select("#", "V", "N", "date", "id", "uid",
                          sdf.power[0].alias("H0"), sdf.power[1].alias(
                              "H1"), sdf.power[2].alias("H2"), sdf.power[3].alias("H3"),
                          sdf.power[4].alias("H4"), sdf.power[5].alias(
                              "H5"), sdf.power[6].alias("H6"), sdf.power[7].alias("H7"),
                          sdf.power[8].alias("H8"), sdf.power[9].alias(
                              "H9"), sdf.power[10].alias("H10"), sdf.power[11].alias("H11"),
                          sdf.power[12].alias("H12"), sdf.power[13].alias(
                              "H13"), sdf.power[14].alias("H14"), sdf.power[15].alias("H15"),
                          sdf.power[16].alias("H16"), sdf.power[17].alias(
                              "H17"), sdf.power[18].alias("H18"), sdf.power[19].alias("H19"),
                          sdf.power[20].alias("H20"), sdf.power[21].alias(
                              "H21"), sdf.power[22].alias("H22"), sdf.power[23].alias("H23"),
                          sdf.statistics[0].alias("S0"), sdf.statistics[1].alias("S1"), sdf.statistics[2].alias("S2"), sdf.statistics[3].alias("S3"))
        return temp

    # call the split_power function
    temp = split(temp)

    # boolean to string (for "N" column)
    temp = temp.withColumn("N", f.col("N").cast('string'))
    # temp.printSchema()

    # filter date
    # temp=temp.filter(temp.date > "2014-08-15").filter(temp.date < "2014-08-19") #filter dates
    # temp=temp.filter(temp.id == "Apt40") #filter IDs
    temp = temp.filter(temp.V == True)  # filter valid rows

    FEATURES = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11',
                'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'S0', 'S1', 'S2', 'S3']

    # call the generate_uniqe_id function
    # temp=generate_uniqe_id(temp)

    # make features ready
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    output = assembler.transform(temp)
    #df_kmeans = vecAssembler.transform(temp).select(col("uid").alias("id"), col("features"))

    # make label ready
    indexer = StringIndexer(inputCol="N", outputCol="label")
    output_fixed = indexer.fit(output).transform(output)

    final_data = output_fixed.select("features", 'label')
    return final_data


# In[15]:


# run mlp method
def mlp(train_data, test_data, layers=[28, 50, 10, 2]):
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(
        maxIter=100, layers=layers, blockSize=128, seed=1234)

    # train_data.show()
    # train the model
    model = trainer.fit(train_data)

    # compute accuracy on the test set
    result = model.transform(test_data)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(
        metricName="accuracy")  # BinaryClassificationEvaluator
    acc = evaluator.evaluate(predictionAndLabels)
    print("Test set accuracy = " + str(acc))
    return acc


# In[ ]:


# In[ ]:
