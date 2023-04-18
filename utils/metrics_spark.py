# utils/metrics_spark.py
# 第三方库
from pyspark.ml.evaluation import RegressionEvaluator


def all_metrics_spark(predictions, return_metrics=False):
    """返回或打印全部误差评价指标 Spark

    参数:
        predictions (spark dataframe): 包含观测值/真值和预测值的dataframe.
        return_metrics (bool, optional): 是否返回指标变量. 默认为 False.

    返回值:
        dict: 由全部误差评价指标构成的字典
    """

    # 构建回归评价器
    evaluator = RegressionEvaluator(
        predictionCol="prediction", labelCol="label")

    # 模型评价
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }

    # 输出结果
    if return_metrics:
        return metrics
    else:
        print(f"mse={metrics['mse']:.3f}")
        print(f"rmse={metrics['rmse']:.3f}")
        print(f"mae={metrics['mae']:.3f}")
        print(f"r2={metrics['r2']:.3f}")
