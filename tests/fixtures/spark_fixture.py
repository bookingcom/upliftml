import logging
import random
import tempfile
from typing import Generator

from ai.h2o.sparkling import H2OConf, H2OContext  # type: ignore
import h2o  # type: ignore
import pytest
from pyspark.sql import SparkSession


def _suppress_py4j_logging() -> None:
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)


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
    spark = _create_integration_test_pyspark_session(hive_warehouse_dir, derby_tmp_dir, h2o_logs_dir)
    spark.sql("DROP DATABASE IF EXISTS test CASCADE")
    spark.sql("CREATE DATABASE IF NOT EXISTS test")
    spark.sql("USE test")
    yield spark
    spark.stop()
