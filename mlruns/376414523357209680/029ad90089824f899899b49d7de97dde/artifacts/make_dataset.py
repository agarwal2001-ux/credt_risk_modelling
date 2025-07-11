import pandas as pd
import numpy as np
import os
import logging

# ---------------- Logger Setup ----------------
logger = logging.getLogger("make_dataset")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

file = logging.FileHandler("make_dataset.log")  # Added .log extension
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)

# ---------------- MLflow Setup ----------------
import mlflow
mlflow.set_experiment("setup_make_dataset")
mlflow.autolog()



# ---------------- Function Definitions ----------------
def main():
    with mlflow.start_run():
        def data(df1_path, df2_path): 
            try:
                df1 = pd.read_csv(df1_path)
                df2 = pd.read_csv(df2_path)
                logger.info("df1, df2 are read successfully")
                return df1, df2
            except Exception as e: 
                logger.error(f"Failed to read the dataframes: {e}")
                raise

        def replace(df1, df2):
            try:
                df1 = df1.replace(-99999, np.nan)
                df2 = df2.replace(-99999, np.nan)
                df1.columns = df1.columns.str.lower()
                df2.columns = df2.columns.str.lower()
                logger.info("Both dataframes have replaced -99999 and lowered column names")
                return df1, df2
            except Exception as e: 
                logger.error(f"Replacement error: {e}") 
                raise

        def merge(df1, df2):
            try:
                new_df = df1.merge(df2, on="prospectid", how="inner")
                logger.info("Data merged successfully")
                return new_df
            except Exception as e: 
                logger.error(f"Merging error occurred: {e}")
                raise

        def save(new_df, path_r):
            try:       
                os.makedirs(path_r, exist_ok=True)
                new_df.to_csv(os.path.join(path_r,"merged.csv"), index=False)
                logger.info("Saved successfully to %s", path_r)
            except Exception as e:
                logger.error(f"Error in saving the dataset: {e}")
                raise



        mlflow.log_artifact(__file__)
        # ---------------- Main Script ----------------
        
        df1, df2 = data(os.path.join("data", "raw", "case_study1.csv"),os.path.join("data", "raw", "case_study2.csv"))
        df1, df2 = replace(df1, df2)
        new_df = merge(df1, df2)
        save(new_df, os.path.join("data", "interim"))

    # Entry point
if __name__ == "__main__":
    main()
