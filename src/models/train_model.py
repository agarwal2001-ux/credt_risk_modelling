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
os.makedirs(encoder_path,exist_ok=True)
pipe_path = os.path.join("models","pipeline")
os.makedirs(pipe_path,exist_ok= True)

# logger configuration
logger = logging.getLogger("training")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

file = logging.FileHandler("train_model1.log")  # Added .log extension
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
                dump(lb,os.path.join(path,"label_encoder.joblib"))
                logger.info("encoder saved successfully")
                return ytrain,ytest
            except Exception as e:
                logger.error(f"error occured - {e}")
                raise


        def final_model(xtrain,ytrain,xtest,ytest,path):
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
                        ("model", LGBMClassifier(n_jobs=-1))
                    ])

                    final_pipe.fit(xtrain, ytrain)
                    dump(final_pipe, os.path.join(path, "final_model.joblib"))
                    logger.info("Final trained model saved")

                    pred = final_pipe.predict(xtest)
                    ac_score = accuracy_score(ytest,pred)
                    report_dict = classification_report(ytest, pred, output_dict=True)

            
            except Exception as e:
                logger.error(f"error occured {e}")
                raise

        
        # call functions
        
        df = read(path=os.path.join("data","final_model"))
        xtrain,xtest,ytrain,ytest = splitting(df)
        
        ytrain_transformed, ytest_transformed = encoder(ytrain,ytest,encoder_path)

        # save the best model
        md = final_model(xtrain,ytrain_transformed,xtest,ytest_transformed,pipe_path)



if __name__ == "__main__":
    main()








