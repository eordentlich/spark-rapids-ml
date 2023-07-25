from .sparksession import CleanSparkSession
from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from typing import Tuple, Dict, Any

import numpy as np
import pytest

from .sparksession import CleanSparkSession
from .utils import (
    create_pyspark_dataframe,
    idfn,
    make_classification_dataset,
    array_equal,
    assert_params
)

def test_toy_example(gpu_number: int) -> None:
    # reduce the number of GPUs for toy dataset to avoid empty partition
    gpu_number = min(gpu_number, 2)
    data = [
        ([1., 2.], 1.),
        ([1., 3.], 1.),
        ([2., 1.], 0.),
        ([3., 1.], 0.),
    ]

    with CleanSparkSession() as spark:
        features_col = "features"
        label_col = "label"
        schema = features_col + " array<float>, " + label_col + " float" 
        df = spark.createDataFrame(data, schema=schema)

        lr_estimator = LogisticRegression(num_workers=gpu_number)
        lr_estimator.setFeaturesCol(features_col)
        lr_estimator.setLabelCol(label_col)
        lr_model = lr_estimator.fit(df)

        assert len(lr_model.coef_) == 1 
        assert lr_model.coef_[0] == pytest.approx([-0.71483153, 0.7148315], abs=1e-6)
        assert lr_model.intercept_ == pytest.approx([-2.2614916e-08], abs=1e-6)
        assert lr_model.n_cols == 2
        assert lr_model.dtype == "float32"

        preds_df = lr_model.transform(df)
        preds = [ row["prediction"] for row in preds_df.collect()]
        assert preds == [1., 1., 0., 0.]

def test_params(tmp_path: str) -> None:

   # Default params
    default_spark_params = {
        "maxIter": 100,
        "regParam": 1.0, # TODO: support default value 0.0, i.e. no regularization
        "tol": 1e-06,
        "fitIntercept": True,
    }

    default_cuml_params = {
        "max_iter": 100,
        "C": 1.0,
        "tol": 1e-6,
        "fit_intercept": True,
    }

    default_lr = LogisticRegression()

    assert_params(default_lr, default_spark_params, default_cuml_params)

    # Spark ML Params
    spark_params: Dict[str, Any] = {
        "maxIter": 30,
        "regParam": 0.5,
        "tol": 1e-2,
        "fitIntercept": False,
    }

    spark_lr = LogisticRegression(**spark_params)
    expected_spark_params = default_spark_params.copy()
    expected_spark_params.update(spark_params)
    expected_cuml_params = default_cuml_params.copy()
    expected_cuml_params.update(
        {
            "max_iter": 30,
            "C" : 2.0, # C should be equal to 1 / regParam 
            "tol": 1e-2,
            "fit_intercept": False,
        }
    )
    assert_params(spark_lr, expected_spark_params, expected_cuml_params)

    # Estimator persistence
    path = tmp_path + "/logistic_regression_tests"
    estimator_path = f"{path}/logistic_regression"
    spark_lr.write().overwrite().save(estimator_path)
    loaded_lr = LogisticRegression.load(estimator_path)
    assert_params(loaded_lr, expected_spark_params, expected_cuml_params)

# TODO support float64
# 'vector' will be converted to float64 so It depends on float64 support  
@pytest.mark.parametrize("fit_intercept", [True, False])  
@pytest.mark.parametrize("feature_type", ["array", "multi_cols"])  
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])  
@pytest.mark.parametrize("max_record_batch", [100, 10000]) 
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.slow
def test_classifier(
    fit_intercept: bool,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    n_classes: int,
    gpu_number: int,
) -> None:
    tolerance = 0.001

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=data_type,
        nrows=data_shape[0],
        ncols=data_shape[1],
        n_classes=n_classes,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

    from cuml import LogisticRegression as cuLR
    cu_lr = cuLR(fit_intercept = fit_intercept)
    cu_lr.fit(X_train, y_train)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_lr = LogisticRegression(
            fitIntercept=fit_intercept,
            num_workers=gpu_number,
        )
        spark_lr.setFeaturesCol(features_col)
        spark_lr.setLabelCol(label_col)
        spark_lr_model: LogisticRegressionModel = spark_lr.fit(train_df)

        # test coefficients and intercepts
        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == cu_lr.dtype

        assert array_equal(spark_lr_model.coef_, cu_lr.coef_, tolerance)
        assert array_equal(spark_lr_model.intercept_, cu_lr.intercept_, tolerance)

        # test transform
        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_lr_model.transform(test_df).collect()
        spark_preds = [row["prediction"] for row in result]
        cu_preds = cu_lr.predict(X_test)
        assert array_equal(cu_preds, spark_preds, 1e-3)