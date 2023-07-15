from .sparksession import CleanSparkSession
from spark_rapids_ml.classification import LogisticRegression, LogisticRegressionModel
from typing import Tuple

import numpy as np
import pytest

from .sparksession import CleanSparkSession
from .utils import (
    create_pyspark_dataframe,
    idfn,
    make_classification_dataset,
    array_equal
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


# TODO support float64
# 'vector' will be converted to float64 so It depends on float64 support  
@pytest.mark.parametrize("feature_type", ["array", "multi_cols"])  
@pytest.mark.parametrize("data_shape", [(2000, 8)], ids=idfn)
@pytest.mark.parametrize("data_type", [np.float32])  
@pytest.mark.parametrize("max_record_batch", [100, 10000]) 
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.slow
def test_classifier(
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
    cu_lr = cuLR()
    cu_lr.fit(X_train, y_train)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )

        assert label_col is not None
        spark_lr = LogisticRegression(
            num_workers=gpu_number,
        )
        spark_lr.setFeaturesCol(features_col)
        spark_lr.setLabelCol(label_col)
        spark_lr_model: LogisticRegressionModel = spark_lr.fit(train_df)

        assert spark_lr_model.n_cols == cu_lr.n_cols
        assert spark_lr_model.dtype == cu_lr.dtype
        assert len(spark_lr_model.coef_) == len(cu_lr.coef_)
        for i in range(len(spark_lr_model.coef_)):
            assert spark_lr_model.coef_[i] == pytest.approx(cu_lr.coef_[i], tolerance)
        assert spark_lr_model.intercept_ == pytest.approx(cu_lr.intercept_, tolerance)

        # test transform
        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = spark_lr_model.transform(test_df).collect()
        spark_preds = [row["prediction"] for row in result]
        cu_preds = cu_lr.predict(X_test)
        assert array_equal(cu_preds, spark_preds, 1e-3)