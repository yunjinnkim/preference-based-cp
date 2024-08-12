import random
import numpy as np
import torch
import openml
import mysql.connector

import types
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor
from sklearn.model_selection import train_test_split
from simple_model import ConformalRankingPredictor, ConformalPredictor, ClassifierModel
from sklearn.preprocessing import LabelEncoder

from py_experimenter.database_connector_mysql import DatabaseConnectorMYSQL


# def connect(self):

#     db = mysql.connector.connect(
#         host="xxx",
#         user="xxx",
#         password="xxx",
#         database="jonas_test",
#         ssl_disabled=False,
#     )
#     return db


# def _start_transaction(self, connection, readonly=False):
#     if not readonly:
#         connection.start_transaction()


# DatabaseConnectorMYSQL.connect = connect
# DatabaseConnectorMYSQL._start_transaction = _start_transaction

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def worker(parameters: dict, result_processor: ResultProcessor, custom_config: dict):
    mccv_split = parameters["mccv_split"]
    clf_seed = parameters["clf_seed"]
    random.seed(clf_seed)
    np.random.seed(clf_seed)
    torch.manual_seed(clf_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(clf_seed)
        torch.cuda.manual_seed_all(clf_seed)

    dataset = openml.datasets.get_dataset(parameters["openml_id"])
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Automatically identify categorical and numerical columns
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=mccv_split
    )

    # Encode labels
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Preprocessing for numerical data: Impute missing values, then scale
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data: Impute missing values, then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    match parameters["model"]:
        # case "plnet":
        #     predictor = ConformalRankingPredictor(num_classes=y_train)
        case "random_forest":
            model = RandomForestClassifier(random_state=clf_seed)
            predictor = ConformalPredictor(model=model, alpha=parameters["alpha"])

    X_train = preprocessor.fit_transform(X_train)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    predictor.fit(
        X_train,
        y_train,
        cal_size=parameters["fraction_cal_samples"],
        random_state=clf_seed,
    )

    X_test = preprocessor.transform(X_test)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    y_pred_crisp = predictor.model.predict(X_test)
    y_pred_set = predictor.predict_set(X_test)

    # Classifcation metrics of crisp (point) prediction
    score_f1 = f1_score(y_test, y_pred_crisp, average="macro")
    score_acc = accuracy_score(y_test, y_pred_crisp)
    score_bacc = balanced_accuracy_score(y_test, y_pred_crisp)
    # score_roc_auc = roc_auc_score(
    #     y_test, y_pred_crisp, average="macro", multi_class="ovr"
    # )

    # Conformal prediction metrics
    hits = [y_test[i] in y_pred_set[i] for i in range(len(y_test))]
    lens = [len(y_test[i]) for i in range(len(y_test))]
    coverage_mean = np.mean(hits)
    coverage_std = np.std(hits)
    efficiency_mean = np.mean(lens)
    efficiency_std = np.std(lens)

    result_processor.process_results(
        {
            "score_f1": score_f1,
            "score_acc": score_acc,
            "score_bacc": score_bacc,
            # "score_roc_auc": score_roc_auc,
            "coverage_mean": coverage_mean,
            "coverage_std": coverage_std,
            "efficiency_mean": efficiency_mean,
            "efficiency_std": efficiency_std,
        }
    )


if __name__ == "__main__":

    experimenter = PyExperimenter(
        experiment_configuration_file_path="./experiments/config/config.yml"
    )

    # experimenter.db_connector.connect = types.MethodType(
    #     connect, experimenter.db_connector
    # )

    # experimenter.db_connector._start_transaction = types.MethodType(
    #     _start_transaction, experimenter.db_connector
    # )

    experimenter.fill_table_from_config()

    experimenter.execute(max_experiments=1, experiment_function=worker)
