import cudf
import os
import pandas as pd

from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.metrics import confusion_matrix, accuracy_score
from cuml.model_selection import train_test_split
from joblib import dump, load

from preprocess import preprocessing_pipeline



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir.removesuffix('/model'))
    
    
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None)
    preprocessed_df = cudf.from_pandas(preprocessing_pipeline(df))
    X = preprocessed_df.drop(["attack"], axis=1)
    y = preprocessed_df["attack"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = cuRF()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cu_score = accuracy_score(y_test, y_pred)
    print("accuracy: ", cu_score )

    print(confusion_matrix(y_test, y_pred))
    
    dump(model, "model/NIDS-RF.model")
    