import pandas as pd
import numpy as np
from scipy.stats import f_oneway,chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import mlflow
import yaml
import os
import json

logger = logging.getLogger("features log")
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


console = logging.StreamHandler()
console.setLevel("DEBUG")
console.setFormatter(formatter)

file = logging.FileHandler("features.log")
file.setLevel("DEBUG")

mlflow.set_experiment("build_features")

def main():
    
    with mlflow.start_run():
        def read_params(file):
            try:
                with open(file,"rb") as file:
                    config = yaml.safe_load(file)
                sg = config["build_features"]["significance_level"]
                logger.info("sig level read_successfully")
                mlflow.log_param("significance_level",sg)
                multi_val = config["build_features"]["multi_val"]
                logger.info("multi_val read_successfully")
                mlflow.log_param("vif_val",multi_val)
                return sg,multi_val
            except Exception as e:
                logger.error(f"error occured {e}")
                raise



        def load(path):
            try:
                df = pd.read_csv(os.path.join(path,"imputation.csv"))
                logger.info("df read successfully")
                return df
            except Exception as e:
                logger.error(f"error cause - {e}")
                raise
        
        
        def types(df):
            try:
                numcols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
                numcols.remove("prospectid")
                mlflow.log_param("num_cols",json.dumps(numcols))
                catcols = df.select_dtypes(include=["object","category"]).columns.tolist()
                mlflow.log_param("cat_cols",json.dumps(catcols))
                
                logger.info("cat and num cols seprated")

                numdf = df[numcols]
                logger.info("numdf created successfully")
                return numcols,catcols,numdf
            except Exception as e:
                logger.error(f"error occusered {e}")
                raise



        def anova(df,numcols,numdf,sg):
            try:
                

                columns_to_be_kept_numerical = []

                for i in numcols:
                    a = list(numdf[i])  
                    b = list(df['approved_flag'])  
                    
                    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
                    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
                    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
                    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


                    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

                    if p_value <= sg:
                        columns_to_be_kept_numerical.append(i)

                after_anova = numdf[columns_to_be_kept_numerical]
                logger.info("after anova test, num cols removed successfully")
                mlflow.log_param("num_cols_kept_anova",json.dumps(columns_to_be_kept_numerical))
                return after_anova,columns_to_be_kept_numerical
            except Exception as e:
                logger.error(f"error cause {e}")
                raise


        def multicollinearity(after_anova,columns_to_be_kept_numerical,multi_val):
            try:
                vif_data = after_anova
                total_columns = vif_data.shape[1]
                columns_to_be_kept = []
                vif = []
                column_index = 0

                for i in range (0,total_columns):
                    
                    vif_value = variance_inflation_factor(vif_data, column_index)
                    
                    if vif_value <= multi_val:
                        columns_to_be_kept.append(columns_to_be_kept_numerical[i] )
                        vif.append(vif_value)
                        column_index = column_index+1
                    
                    else:vif_data = vif_data.drop([columns_to_be_kept_numerical[i] ] , axis=1)
                
                logger.info("multicollinearity successful")
                mlflow.log_param("VIF_kept_cols",json.dumps(columns_to_be_kept))
                return columns_to_be_kept
            except Exception as e:
                logger.error(f"error cause {e}")

        def save(df,columns_to_be_kept,catcols,path):
            try:
                ready = df[columns_to_be_kept +catcols]
                os.makedirs(path)
                ready.to_csv(os.path.join(path,"model_build.csv"),index=False)
            except Exception as e:
                logger.error(f"error occured {e}")
                raise
        


        sg,multi_val = read_params("params.yaml")        
        df = load(os.path.join("data","imputed"))
        numcols,catcols,numdf = types(df)
        after_anova,columns_to_be_kept_numerical = anova(df,numcols,numdf,sg)
        columns_to_be_kept = multicollinearity(after_anova,columns_to_be_kept_numerical,multi_val)
        saved = save(df,columns_to_be_kept,catcols,os.path.join("data","final_model"))

if __name__ == "__main__":
    main()





