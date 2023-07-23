import sys
import os
import pickle
import json
import numpy as np
import pandas as pd

from dvclive import Live
from matplotlib import pyplot as plt

from sklearn.metrics import recall_score, precision_score, f1_score , accuracy_score, roc_auc_score

def evaluate(model_path, test_file_name):

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Read test data
    test_data = pd.read_csv(test_file_name)
    
    test_features = test_data.drop('Churn', axis = 1)
    test_labels = test_data['Churn']

    # Make predictions on test data
    predictions_proba = model.predict_proba(test_features)
    test_predictions = model.predict(test_features)

    # Compute metrics
    acc = accuracy_score(test_labels, test_predictions)
    prec = precision_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)
    auc_score = roc_auc_score(test_labels, predictions_proba[:, 1])

    print('Evaluation metrics computed')


    EVAL_PATH = 'results'

    os.makedirs(EVAL_PATH, exist_ok = True)

    output = pd.DataFrame({'Actual' : test_labels, 'Predicted': test_predictions})
    output.to_csv(os.path.join(EVAL_PATH , 'predictions_vs_actuals.csv'))

    # Plot feature importances from the random forest model
    fig, axes = plt.subplots()
    fig.subplots_adjust(left = 0.2, bottom = 0.2, top = 0.95)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    features = test_features.columns
    
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color = 'b', align = 'center')
    
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')

    plt.savefig(os.path.join(EVAL_PATH, 'feature_importance.png'))

    # Log metrics, plots, using DVC Live
    with Live(os.path.join(EVAL_PATH, 'live'), dvcyaml=False) as live:
       
        live.log_metric('Test_accuracy_score', acc)
        live.log_metric('Test_precision_score', prec)
        live.log_metric('Test_recall_score', rec)
        live.log_metric('Test_f1_score', f1)
        live.log_metric('AUC_score', auc_score)

        live.log_sklearn_plot(
            'roc', test_labels.squeeze(), 
            predictions_proba[:, 1], name = 'ROC Curve'
        )
        live.log_sklearn_plot(
            'confusion_matrix', test_labels.squeeze(),
            test_predictions, name = 'Confusion-matrix'
        )
        live.log_sklearn_plot(
            'precision_recall', test_labels.squeeze(), 
            predictions_proba[:, 1], name = 'Precision-Recall Curve'
        )
        
        live.log_image(
            'Feature_importance.png', 
            os.path.join(EVAL_PATH, 'feature_importance.png')
        )


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Please provide the serialized model and the name of the input test data file")
    else:
        model_path = sys.argv[1]
        test_file_name = sys.argv[2]
        
        evaluate(model_path, test_file_name)






