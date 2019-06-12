from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('bank.csv', header = True, inferSchema = True)
print(df.printSchema())


import pandas as pd
pdf=pd.DataFrame(df.take(5), columns=df.columns).transpose()
print(pdf.head(5))

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
print(df.select(numeric_features).describe().toPandas().transpose())
