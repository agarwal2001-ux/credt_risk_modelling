import pandas as pd
import numpy as np
from scipy.stats import f_oneway,chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import mlflow

logger = logging.getLogger("features log")
logger.setLevel("DEBUG")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


console = logging.StreamHandler()
console.setLevel("DEBUG")
console.setFormatter(formatter)

file = logging.FileHandler("features.log")
file.setLevel("DEBUG")

mlflow.set_experiment("knn-imputer")

def main():
    mlflow.autolog()
    
    with mlflow.start_run():
        def load():
            try:
                df = pd.read_csv("modeldf.csv")
                logger.info("df read successfully")
                return df
            except Exception as e:
                logger.error(f"error cause - {e}")
                raise
        
        
        def types(df):
            try:
                numcols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
                numcols.remove("prospectid")
                catcols = df.select_dtypes(include=["object","category"]).columns.tolist()

                numdf = df[numcols]
                return numcols,catcols,numdf
            except Exception as e:
                logger.error(f"error occusered {e}")
                raise



        def anova(df,numcols,numdf):
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

                    if p_value <= 0.05:
                        columns_to_be_kept_numerical.append(i)

                after_anova = numdf[columns_to_be_kept_numerical]
                return after_anova,columns_to_be_kept_numerical
            except Exception as e:
                logger.error(f"error cause {e}")
                raise


        def multicollinearity(after_anova,columns_to_be_kept_numerical):
            try:
                vif_data = after_anova
                total_columns = vif_data.shape[1]
                columns_to_be_kept = []
                vif = []
                column_index = 0

                for i in range (0,total_columns):
                    
                    vif_value = variance_inflation_factor(vif_data, column_index)
                    print (column_index,'---',vif_value)
                    
                    
                    if vif_value <= 6:
                        columns_to_be_kept.append(columns_to_be_kept_numerical[i] )
                        vif.append(vif_value)
                        column_index = column_index+1
                    
                    else:vif_data = vif_data.drop([columns_to_be_kept_numerical[i] ] , axis=1)
                return columns_to_be_kept
            except Exception as e:
                logger.error(f"error cause {e}")

        def save(df,columns_to_be_kept,catcols):
            try:
                ready = df[columns_to_be_kept +catcols]
                ready.to_csv("model_build.csv",index=False)
            except Exception as e:
                logger.error(f"error occured {e}")
                raise
        

        df = load()
        numcols,catcols,numdf = types(df)
        after_anova,columns_to_be_kept_numerical = anova(df,numcols,numdf)
        columns_to_be_kept = multicollinearity(after_anova,columns_to_be_kept_numerical)
        saved = save(df,columns_to_be_kept,catcols)

if __name__ == "__main__":
    main()
