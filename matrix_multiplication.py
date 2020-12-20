# import required libraries
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *

import numpy as np

# initiate the spark session
spark = SparkSession \
    .builder \
    .master('yarn') \
    .appName('matrix-multiplication') \
    .enableHiveSupport() \
    .getOrCreate()

# create the guest feature matrix (an IndexedRowMatrix)
# first column in the matrix is composed of the dummy values
# corresponding to the actual row indices
num_guests = 1000000
guest_feature_vector_length = 100

guest_feature_rows = spark.sparkContext.parallelize(range(num_guests), 100) \
    .map(lambda guest_id: IndexedRow(
        guest_id,
        [guest_id] + list(np.random.randint(low = 1, high = 10, size = guest_feature_vector_length))
    ))

guest_feature_matrix = IndexedRowMatrix(guest_feature_rows.repartition(200))
print(f'dimensions of guest feature matrix: {guest_feature_matrix.numRows()} x {guest_feature_matrix.numCols()}')

# create the item feature matrix (a local LocalMatrix)
# first column and first row in the matrix is composed of the dummy values
num_items = 1000
item_feature_vector_length = 100

# first row
item_feature_array = [1] + [0] * item_feature_vector_length
# rest of the rows
for i in range(num_items):
    item_feature_array += [0] + list(np.random.randint(low = 1, high = 10, size = item_feature_vector_length))
item_feature_matrix = Matrices.dense(item_feature_vector_length + 1, num_items + 1, item_feature_array)
print(f'dimensions of item feature matrix: {item_feature_vector_length + 1} x {num_items + 1}')

# calculate the guest-item rating matrix by multiplying
# the guest feature matrix with item feature matrix
ratings_matrix = guest_feature_matrix.multiply(item_feature_matrix)
print(f'dimensions of ratings matrix: {ratings_matrix.numRows()} x {ratings_matrix.numCols()}')

# extract the guest rating vectors
ratings_rdd = ratings_matrix.rows \
    .repartition(500) \
    .map(lambda ele: (ele.index, ele.vector.toArray().tolist())) \
    .map(lambda ele: (int(ele[0]), int(ele[1][0]), ele[1][1:]))

schema = StructType([
    StructField('raw_guest_index', IntegerType(), True),
    StructField('dummy_guest_index', IntegerType(), True),
    StructField('guest_rating_array', ArrayType(DoubleType()), True)
])
ratings = spark.createDataFrame(ratings_rdd, schema)
ratings.filter(col('raw_guest_index') != col('dummy_guest_index')).sample(0.1).show(10)
