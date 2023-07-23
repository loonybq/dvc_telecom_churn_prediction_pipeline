import os
import pickle
import sys
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier

def train(data_file_name):

    # Parameters
    params = yaml.safe_load(open('params.yaml'))

    n_estimators = params['train']['n_estimators']
    max_depth = params['train']['max_depth']
    min_samples_split = params['train']['min_samples_split']
    min_samples_leaf = params['train']['min_samples_leaf']
    class_weight = params['train']['class_weight']

    # Load training data
    train_data = pd.read_csv(data_file_name)

    x_train = train_data.drop(columns = 'Churn')
    y_train = train_data['Churn']

    # Model training 
    model = RandomForestClassifier(
        n_estimators = n_estimators,max_depth = max_depth, 
        min_samples_split = min_samples_split,
        min_samples_leaf = min_samples_leaf, 
        class_weight = class_weight
    )

    model.fit(x_train, y_train)

    print('Training done')

    # Write serialized model
    os.makedirs('model_dir', exist_ok = True)

    model_path = os.path.join('model_dir', 'model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the input training data file")
    else:
        data_file_name = sys.argv[1]
        
        train(data_file_name)












