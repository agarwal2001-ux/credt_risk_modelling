import pandas as pd
import numpy as np
import os
import logging
import yaml
import mlflow
from sklearn.impute import KNNImputer

# ---------------- Logger Setup ----------------
logger = logging.getLogger("imputation")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

file = logging.FileHandler("imputation.log")
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)

# ---------------- MLflow Setup ----------------
mlflow.set_experiment("knn-imputation")

def main():
    with mlflow.start_run():
        
        def read_yaml(file):
            try:
                with open(file, "r") as stream:
                    config = yaml.safe_load(stream)
                    k_neighbours = config["imputation"]["n_neighbours"]
                    weights = config["imputation"]["weights"]
                    logger.info("k_neighbours and weights fetched successfully from YAML")
                    mlflow.log_param("k_neighbours", k_neighbours)
                    mlflow.log_param("weights", weights)
                    return k_neighbours, weights
            except Exception as e:
                logger.error(f"Some error occurred with YAML function: {e}")
                raise 

        def data(path):
            try:
                df = pd.read_csv(os.path.join(path, "merged.csv"))
                logger.info("DataFrame retrieved successfully")
                return df
            except Exception as e:
                logger.error(f"Data cannot be read: {e}")
                raise

        def separation(df):
            try:
                numcols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                numcols.remove("prospectid")
                catcols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                catcols.append("prospectid")
                numdf = df[numcols]
                catdf = df[catcols]
                logger.info("Separation is successful")
                return numdf, catdf, numcols, catcols
            except Exception as e:
                logger.error(f"Separation error: {e}")
                raise

        def imputation(numdf, catdf, numcols, catcols, df, k_neighbours, weights):
            try:
                train = numdf.to_numpy()
                knn = KNNImputer(n_neighbors=k_neighbours, weights=weights)
                new_xtrain = knn.fit_transform(train)
                imputed_df = pd.DataFrame(new_xtrain, columns=numcols)
                imputed_df.insert(loc=0, column="prospectid", value=df["prospectid"].values)
                model_df = imputed_df.merge(catdf, on="prospectid", how="inner")
                logger.info("Imputation is successful")
                return model_df
            except Exception as e:
                logger.error(f"Error occurred during imputation: {e}")
                raise

        def save(model_df, path):
            try:
                os.makedirs(path, exist_ok=True)
                output_path = os.path.join(path, "imputation.csv")
                model_df.to_csv(output_path, index=False)
                logger.info(f"model_df is saved to {output_path}")
                mlflow.log_artifact(output_path)
            except Exception as e:
                logger.error(f"Saving error occurred: {e}")
                raise

        # ---- Pipeline Execution ----
        k_neighbours, weights = read_yaml("params.yaml")
        df = data(os.path.join("data", "interim"))
        numdf, catdf, numcols, catcols = separation(df)
        model_df = imputation(numdf, catdf, numcols, catcols, df, k_neighbours, weights)
        save(model_df, os.path.join("data", "imputed"))

if __name__ == "__main__":
    main()
