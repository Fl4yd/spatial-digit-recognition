import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SimpleTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        pass

    def predict(self,X):
        pass
    
class SimpleRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        pass
    
    def predict(self,X):
        pass



def main():
    # Load the data
    df = pd.read_csv('training_data/combined_strokes.csv')

    # Inspect the data
    #print(df.head())

if __name__ == "__main__":
    main()

