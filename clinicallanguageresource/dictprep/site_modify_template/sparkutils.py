import os

from pyspark.sql import SparkSession


def setup_spark_session(name: str) -> SparkSession:
    """
    Gets a Spark Session. Overload with own implementation if needed to suit local environment.
    """
    spark = SparkSession.builder \
        .config("spark.app.name", name) \
        .config("spark.storage.memoryFraction", "0.02") \
        .config("spark.shuffle.memoryFraction", "0.8") \
        .enableHiveSupport() \
        .getOrCreate()

    # Prevent filesystem perm issues by using current working directory
    os.environ['TRANSFORMERS_CACHE'] = os.getcwd() + "/tmp"
    os.environ['XDG_CACHE_HOME'] = os.getcwd() + "/tmp"
    # Mark transformers library to be offline (assuming we will supply the models as part of the bundle itself)
    # Useful for clusters behind a locked-down firewall
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    return spark
