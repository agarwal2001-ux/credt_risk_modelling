import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import KFold,cross_val_score,StratifiedKFold
import logging
import mlflow
from joblib import dump, load
import os
from lightgbm import LGBMClassifier
import optuna
from optuna.integration.mlflow import MLflowCallback
import json

encoder_path = os.path.join("models","encoder")
os.makedirs(encoder_path)
pipe_path = os.path.join("models","pipeline")
os.makedirs(pipe_path)

# logger configuration
logger = logging.getLogger("training")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

file = logging.FileHandler("train_model.log")  # Added .log extension
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)


# configuration of mlflow
mlflow.set_experiment("training")

def main():
    mlflow.autolog()

    with mlflow.start_run():
        def read(path):
            try:
                df = pd.read_csv(os.path.join(path,"model_build.csv"))
                logger.info("model build read successful")
                return df
            except Exception as e:
                logger.error(f"error occured - {e}")
                raise

        def splitting(df):
            try:
                xtrain,xtest,ytrain,ytest = train_test_split(df.iloc[:,0:45],df.iloc[:,-1],test_size=0.2,random_state=42)
                logger.info("train_test split successful")
                return xtrain,xtest,ytrain,ytest
            except Exception as e:
                logger.error(f"error occured {e}")
                raise

        def encoder(ytrain,ytest,path):
            try:
                lb = LabelEncoder()
                ytrain = lb.fit_transform(ytrain)
                ytest = lb.transform(ytest)
                logger.info("ytrain and ytest transformed")
                os.makedirs("references")
                dump(lb,os.path.join(path,"label_encoder.joblib"))
                logger.info("encoder saved successfully")
                return ytrain,ytest
            except Exception as e:
                logger.error(f"error occured - {e}")
                raise

        def find_parameters(xtrain,ytrain,trial):
            try:
                    transformer = ColumnTransformer(
                        [("ordinal",OrdinalEncoder(categories=[["SSC","12TH","UNDER GRADUATE","GRADUATE","POST-GRADUATE","PROFESSIONAL","OTHERS"]],
                        handle_unknown="use_encoded_value",
                        unknown_value= -1,
                        encoded_missing_value=np.nan),["education"]),
                        ("onehot",OneHotEncoder(drop="first",sparse_output=False),["maritalstatus","gender","last_prod_enq2","first_prod_enq2"])
                        ]
                        , remainder="passthrough")


                    n_estimators = trial.suggest_int("model__n_estimators",10,1000)
                    max_depth = trial.suggest_int("model__max_depth",2,10)
                    subsample = trial.suggest_float("model__subsample",0.1,0.9)
                    learning_rate = trial.suggest_float("model__learning_rate",0.001,1)
                    reg_lambda = trial.suggest_int("model__reg_lambda",1,1000)
                    colsample_bylevel = trial.suggest_float("model__colsample_bylevel",0.2,0.8)
                    grow_policy = trial.suggest_categorical("model__grow_policy",["depthwise","lossguide"])
                    min_child_weight = trial.suggest_float("model__min_child_weight",0.1,1)


                    pipe2 = Pipeline([
                    ("pre-processing",transformer),
                    ("model",LGBMClassifier(n_jobs=-1,n_estimators=n_estimators,max_depth=max_depth,
                                        subsample=subsample,learning_rate=learning_rate,reg_lambda=reg_lambda,
                                        colsample_bylevel=colsample_bylevel))
                    ])
                    sf = StratifiedKFold(n_splits=5,shuffle=True)
                    score = cross_val_score(pipe2,xtrain,ytrain,cv=sf,scoring="accuracy").mean()

                    return score
                
            except Exception as e:
                logger.error(f"error occured {e}")
                raise
        

        def final_model(xtrain,ytrain_transformed,best_params,xtest,ytest,path):
            try:
                final_transformer = ColumnTransformer([
                ("ordinal", OrdinalEncoder(
                    categories=[["SSC", "12TH", "UNDER GRADUATE", "GRADUATE", "POST-GRADUATE", "PROFESSIONAL", "OTHERS"]],
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan), ["education"]),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False),
                ["maritalstatus", "gender", "last_prod_enq2", "first_prod_enq2"])
                ], remainder="passthrough")

                final_pipe = Pipeline([
                    ("pre-processing", final_transformer),
                    ("model", LGBMClassifier(n_jobs=-1, **best_params))
                ])

                final_pipe.fit(xtrain, ytrain_transformed)
                dump(final_pipe, os.path.join(path, "final_model.joblib"))
                logger.info("Final trained model saved")

                pred = final_pipe.predict(xtest)
                ac_score = accuracy_score(ytest,pred)
                report_dict = classification_report(ytest, pred, output_dict=True)
                for label, metrics in report_dict.items():
                    if isinstance(metrics, dict):
                        for metric, val in metrics.items():
                            mlflow.log_metric(f"{label}_{metric}", val)
                mlflow.log_metric("accuracy_score", ac_score)

                mlflow.log_metric("ac_score",ac_score)

            
            except Exception as e:
                logger.error(f"error occured {e}")
                raise

        
        # call functions
        
        df = read(path=os.path.join("data","final_model"))
        xtrain,xtest,ytrain,ytest = splitting(df)
        
        ytrain_transformed, ytest_transformed = encoder(ytrain,ytest,encoder_path)
        
        # mlflow with optuna integration
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="accuracy"
        )
        def objective(trial):
            return find_parameters(xtrain, ytrain_transformed, trial)
        
        study = optuna.create_study(direction="maximize", study_name="credit_risk_model")
        study.optimize(objective, n_trials=30, callbacks=[mlflow_callback])

        best_params = study.best_params
        dump(best_params, os.path.join("references","best_params.joblib"))
        logger.info(f"Best parameters: {best_params}")

        # save the best model
        md = final_model(xtrain,ytrain_transformed,best_params,xtest,ytest,pipe_path)



if __file__ == "__main__":
    main()








