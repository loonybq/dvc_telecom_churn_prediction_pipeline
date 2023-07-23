import os
import pandas as pd
import yaml
import sys
from sklearn.model_selection import train_test_split


def process_data(data_file_name):

	# Parameters
	split_ratio = yaml.safe_load(open('params.yaml'))['data_process']['split_ratio']

	churn_data = pd.read_csv(data_file_name)

	print('Data shape: ', churn_data.shape)
	
	print('Null values for each feature:\n', churn_data.isnull().sum())

	churn_data_cleaned = churn_data.fillna(churn_data.mean())

	print('Sum of Null values for each feature after processing:', 
		sum(churn_data_cleaned.isnull().sum()))
	
	# Train test split 
	train, test = train_test_split(churn_data_cleaned , test_size = split_ratio)

	# Create folder to save file
	data_path = 'processed_data'
	os.makedirs(data_path, exist_ok = True)

	# Save prepared data
	train.to_csv(os.path.join(data_path, 'out_train.csv'), index = False)
	test.to_csv(os.path.join(data_path, 'out_test.csv'), index = False)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the input data file")
    else:
        data_file_name = sys.argv[1]
        
        process_data(data_file_name)

