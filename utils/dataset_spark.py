# utils/dataset_spark.py
# 第三方库
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lag
from pyspark.sql.functions import col


def moving_window(seriesDF, n_in, name_features):
    """将时间序列数据转换为监督学习数据集 Spark

    参数:
        seriesDF (spark dataframe): 时间序列数据.
        n_in (int): 输入变量X的滞后数目.
        name_features (list of strings): 输入变量X的名称列表

    返回值:
        spark dataframe: 监督学习数据集
    """

    # 窗口变量
    w = Window().partitionBy().orderBy(col('id'))
    # 全部特征名称
    all_column_names = []

    # 遍历各站点
    for i in range(len(name_features)):
        # 特征名称
        column_names = [f'var{i}(t-{j+1})' for j in range(n_in)]
        all_column_names.extend(column_names)
        # 滑动窗口
        for k in range(n_in):
            seriesDF = seriesDF.withColumn(
                column_names[k], lag(name_features[i], (k+1), 0).over(w))
        seriesDF = seriesDF.withColumnRenamed(name_features[i], f'var{i}')

    # 组合特征向量
    assembler = VectorAssembler(
        inputCols=all_column_names, outputCol='features')
    seriesDF = assembler.transform(seriesDF)

    return seriesDF
