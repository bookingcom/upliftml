import logging
import os
import random
import tempfile
from typing import Generator

import h2o  # type: ignore
import pytest
from ai.h2o.sparkling import H2OConf, H2OContext  # type: ignore
from pyspark.sql import SparkSession


def _suppress_py4j_logging() -> None:
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)


def _verify_spark(spark: SparkSession) -> None:
    """We need to verify that spark is running locally, without
    access to any system-level configured cluster. A proxy for
    that is the absence of a hive.metastore.uris property"""

    # noinspection PyProtectedMember
    iterator = spark.sparkContext._jsc.hadoopConfiguration().iterator()  # type: ignore
    for conf in iterator:
        if conf.getKey() == "hive.metastore.uris":
            raise ValueError(
                f"Detected a hive metastore configuration: {conf}. "
                "Most likely this means this spark session is connected "
                "to a remote cluster"
            )


def _clean_system_spark_env() -> None:
    # The py/spark environment variables, if set, prevent
    # from running the local test pyspark installation
    # and run the system one. We never restore the
    # previously set values (if there were any).
    os.environ.pop("SPARK_HOME", None)
    os.environ.pop("PYSPARK_PYTHON", None)


def _create_h2o_context() -> None:
    conf = H2OConf().setInternalClusterMode()
    H2OContext.getOrCreate(conf)
    h2o.no_progress()


def _create_integration_test_pyspark_session(
    warehouse_dir: tempfile.TemporaryDirectory,
    derby_dir: tempfile.TemporaryDirectory,
    h2o_logs_dir: tempfile.TemporaryDirectory,
) -> SparkSession:
    # Choose a random port to avoid clashing if many Spark instances run concurrently
    random_port = 4040 + random.randint(1, 100)
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("local-pyspark-test")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.port", str(random_port))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", warehouse_dir.name)
        .config("spark.driver.extraJavaOptions", "-Dderby.system.home=%s" % derby_dir.name)
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.codegen.wholeStage", "false")
        .config("spark.ext.h2o.log.level", "FATAL")
        .config("spark.ext.h2o.node.log.dir", h2o_logs_dir.name)
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Create H2O context
    _create_h2o_context()

    return spark


@pytest.fixture(scope="session")
def derby_tmp_dir() -> Generator[tempfile.TemporaryDirectory, None, None]:
    tmp = tempfile.TemporaryDirectory("_b_uplift_derby", "_test")
    yield tmp
    tmp.cleanup()


@pytest.fixture(scope="session")
def hive_warehouse_dir() -> Generator[tempfile.TemporaryDirectory, None, None]:
    tmp = tempfile.TemporaryDirectory("_b_uplift_warehouse", "_test")
    yield tmp
    tmp.cleanup()


@pytest.fixture(scope="session")
def h2o_logs_dir() -> Generator[tempfile.TemporaryDirectory, None, None]:
    tmp = tempfile.TemporaryDirectory("_b_uplift_h2o_logs", "_test")
    yield tmp
    tmp.cleanup()


@pytest.fixture(scope="session")
def spark(
    hive_warehouse_dir: tempfile.TemporaryDirectory,
    derby_tmp_dir: tempfile.TemporaryDirectory,
    h2o_logs_dir: tempfile.TemporaryDirectory,
) -> Generator[SparkSession, None, None]:
    _suppress_py4j_logging()
    _clean_system_spark_env()
    spark = _create_integration_test_pyspark_session(hive_warehouse_dir, derby_tmp_dir, h2o_logs_dir)
    _verify_spark(spark)
    spark.sql("DROP DATABASE IF EXISTS test CASCADE")
    spark.sql("CREATE DATABASE IF NOT EXISTS test")
    spark.sql("USE test")
    yield spark
    spark.stop()
