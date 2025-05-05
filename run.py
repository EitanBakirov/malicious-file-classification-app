import pandas as pd
import numpy as np

import scipy.stats as stats

import sklearn as skl
import seaborn as sns

from matplotlib import pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.ensemble import IsolationForest

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from IPython.display import display

import time

import warnings
warnings.filterwarnings('ignore')


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Add at the top of run.py
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


def fill_corr_missing_values(data, corr_matrix):
    
    # Iterate over the columns with missing values
    for column in data.columns[data.isnull().any()]:
        if column == 'size':
            correlated_features = [('numstrings', corr_matrix.loc['size', 'numstrings']),
                                   ('MZ', corr_matrix.loc['size', 'MZ'])]
        elif column == 'numstrings':
            correlated_features = [('size', corr_matrix.loc['size', 'numstrings']),
                                   ('MZ', corr_matrix.loc['numstrings', 'MZ'])]
        elif column == 'MZ':
            correlated_features = [('size', corr_matrix.loc['size', 'MZ']),
                                   ('numstrings', corr_matrix.loc['numstrings', 'MZ'])]
        else:
            correlated_features = []

        if correlated_features:
            # Iterate over the rows with null values in the current column
            for index, row in data[data[column].isnull()].iterrows():
                for feature, correlation_value in correlated_features:
                    if not pd.isnull(row[feature]):
                        expected_value = correlation_value * row[feature]
                        data.loc[index, column] = expected_value
                        break

    return data


def handle_corr_missing_values(train_data, test_data):
    # Calculate correlation matrix
    corr_matrix = train_data[['size', 'numstrings', 'MZ']].corr()
    train_data = fill_corr_missing_values(train_data, corr_matrix)
    test_data = fill_corr_missing_values(test_data, corr_matrix)

    return train_data, test_data


def calc_median_value(df, numeric_features):
    medians = {}
    for column in numeric_features:
        median_value = df[column].median()
        medians[column] = median_value
    return medians


def calc_majority(df, features):
    majority_values = {}
    for feature in features:
        mode_value = df[feature].value_counts().idxmax()
        majority_values[feature] = mode_value
    return majority_values


def fill_rest_missing_values(data, columns_to_impute, medians, majority_values):

    for column in columns_to_impute:
        if data[column].dtype == 'object':  # Categorical column
            data[column].fillna(majority_values[column], inplace=True)
        elif data[column].dtype in ['float64', 'int64']:  # Float64 and Int64 column
            data[column].fillna(medians[column], inplace=True)

    return data


def handle_rest_missing_values(train_data, test_data):
    numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    category_features = train_data.select_dtypes(include=['object']).columns.tolist()

    medians = calc_median_value(train_data, numeric_features)
    majority_values = calc_majority(train_data, category_features)

    # columns_to_impute = ['size', 'vsize', 'imports', 'exports', 'has_debug', 'has_tls', 'has_resources', 'has_relocations', 'has_signature', 'symbols', 'numstrings', 'avlength', 'printables', 'paths', 'urls', 'registry', 'MZ', 'file_type_trid', 'file_type_prob_trid', 'A', 'B', 'C']
    columns_to_impute_train = train_data.columns
    columns_to_impute_test = test_data.columns
    train_data = fill_rest_missing_values(train_data, columns_to_impute_train, medians, majority_values)
    test_data = fill_rest_missing_values(test_data, columns_to_impute_test, medians, majority_values)

    return train_data, test_data


def handle_missing_values(train_data, test_data):
    train_data, test_data = handle_corr_missing_values(train_data, test_data)
    train_data, test_data = handle_rest_missing_values(train_data, test_data)
    return train_data, test_data


def adding_new_feature(data):
    # Calculate the total number of functions (imports + exports)
    total_functions = data['imports'] + data['exports']

    # Calculate the proportion of imported functions, handling the case when total_functions is zero
    proportion_imports = data['imports'] / np.where(total_functions == 0, 1, total_functions)

    # Create a new column for the proportion of imported functions
    data['proportion_imports'] = proportion_imports

    return data


def reducing_category_types(data):
    threshold = 0.01  # Set the threshold to 1% of the observations

    # Calculate the value counts of each category
    category_counts = data['file_type_trid'].value_counts(normalize=True)

    # Identify the categories that appear in less than the threshold
    categories_to_change = category_counts[category_counts < threshold].index.tolist()

    # Replace the categories with 'other'
    data.loc[data['file_type_trid'].isin(categories_to_change), 'file_type_trid'] = 'other'

    # Get the remaining categories in the data
    remaining_categories = data['file_type_trid'].unique().tolist()

    return data, remaining_categories



def apply_remaining_categories_for_test(data, remaining_categories):
    
    # Replace categories not in remaining_categories with 'other'
    data.loc[~data['file_type_trid'].isin(remaining_categories), 'file_type_trid'] = 'other'

    return data


def categorical_variables_spread(train_data, test_data):
    categorical_columns = ['file_type_trid', 'C']

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit the encoder on the train data
    encoder.fit(train_data[categorical_columns])

    # Transform both train and test data
    train_encoded = encoder.transform(train_data[categorical_columns])
    test_encoded = encoder.transform(test_data[categorical_columns])

    # Create custom feature names for the encoded columns
    feature_names = []
    for i, column in enumerate(categorical_columns):
        for feature in encoder.categories_[i]:
            feature_names.append(f'{column}_{feature}')

    # Create DataFrames with the encoded columns and custom feature names
    train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_data.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_data.index)

    # Concatenate the encoded columns with the original DataFrames
    train_data = pd.concat([train_data.reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
    test_data = pd.concat([test_data.reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

    # Drop the original categorical columns
    train_data.drop(columns=categorical_columns, inplace=True)
    test_data.drop(columns=categorical_columns, inplace=True)

    return train_data, test_data


def handle_categorial_features(train_data, test_data):
    train_data, remaining_categories = reducing_category_types(train_data)
    test_data = apply_remaining_categories_for_test(test_data, remaining_categories)

    # Use the updated categorical_variables_spread function
    train_data, test_data = categorical_variables_spread(train_data, test_data)

    return train_data, test_data

def apply_log(data, log_features):
    data[log_features] = np.log(data[log_features])
    return data

def identify_bounds(df, feature, lower_percentile = 0.25, upper_percentile = 0.75, threshold=1.5):
    # Calculate the specified percentiles
    p1 = df[feature].quantile(lower_percentile)
    p2 = df[feature].quantile(upper_percentile)
    spread = p2 - p1

    # Define the upper and lower bounds
    lower_bound = p1 - threshold * spread
    upper_bound = p2 + threshold * spread

    # print(f"For {feature} the threshold given {threshold}, Lower Bound: '{lower_bound}', Upper Bound: '{upper_bound}'.")
    return lower_bound, upper_bound

def handle_ND_outliers_train(data):
    log_features = ['size', 'vsize', 'numstrings', 'printables']
    data = apply_log(data, log_features)

    normal_features = ['size', 'vsize', 'numstrings', 'printables', 'A'] 
    bounds = {}
    
    for feature in normal_features:
        lower_bound, upper_bound = identify_bounds(data, feature)
        
        data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])
        data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])
        
        bounds[feature] = (lower_bound, upper_bound)

    return data, bounds

def handle_ND_outliers_test(data, feature_bounds):

    log_features = ['size', 'vsize', 'numstrings', 'printables']
    data = apply_log(data, log_features)

    for feature, (lower_bound, upper_bound) in feature_bounds.items():
        data[feature] = np.where(data[feature] < lower_bound, lower_bound, data[feature])
        data[feature] = np.where(data[feature] > upper_bound, upper_bound, data[feature])

    return data

def handle_NND_outliers_train(data, labels, threshold, contamination):
    
    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42) 

    non_norm_dist_features = ['imports', 'exports', 'symbols', 'paths', 'urls', 'registry', 'MZ', 'avlength', 'proportion_imports']
    data_selected = data[non_norm_dist_features].copy()

    model.fit(data_selected)

    outlier_scores = model.decision_function(data_selected)
    outlier_indices = np.where(outlier_scores < threshold)[0]

    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    
    # Remove the outlier indices from the data
    data = data.drop(outlier_indices)
    labels = labels.drop(outlier_indices)

    return data, labels

def handle_outliers(train_data, train_labels, test_data):

    train_data, bounds = handle_ND_outliers_train(train_data)
    test_data = handle_ND_outliers_test(test_data, bounds)

    contamination = 0.001
    threshold = 0.001
    train_data, train_labels = handle_NND_outliers_train(train_data, train_labels, threshold, contamination)

    return train_data, train_labels, test_data

def normalize_data_train(data, selected_features):
    
    # Filter the data to keep only the selected features
    selected_data = data[selected_features]

    # Standardize the selected_data using StandardScaler
    scaler = StandardScaler()
    scaler = scaler.fit(selected_data)
    scaled_data = scaler.transform(selected_data)

    # Create a new DataFrame with the scaled data and original column names
    data[selected_features] = scaled_data
    
    return data, scaler

def normalize_data_test(data, scaler, selected_features):
    
    # Filter the data to keep only the selected features
    selected_data = data[selected_features]

    # Standardize the selected_data using the fitted scaler by the train data
    scaled_data = scaler.transform(selected_data)

    # Create a new DataFrame with the scaled data and original column names
    data[selected_features] = scaled_data

    return data


def normalize_data(train_data, test_data):

    selected_features = ['size', 'vsize', 'numstrings', 'printables', 'A','imports', 'exports', 'symbols', 'paths', 'urls', 'registry', 'MZ', 'avlength', 'proportion_imports', 'file_type_prob_trid', 'B']

    train_data, scaler = normalize_data_train(train_data, selected_features)
    test_data = normalize_data_test(test_data, scaler, selected_features)

    return train_data, test_data

def remove_features(data):
    features_to_remove = ['registry', 'size', 'MZ']
    data_selected = data.drop(features_to_remove, axis=1)
    return data_selected

def preprocess_data_ALL(train_data, train_labels, test_data):

    if 'sha256' in train_data.columns:
        # Drop 'sha256' column
        train_data = train_data.drop('sha256', axis=1)

    if 'sha256' in test_data.columns:
        # Drop 'sha256' column
        test_data = test_data.drop('sha256', axis=1)

    # Missing Values Handling
    train_data, test_data = handle_missing_values(train_data, test_data)
    
    # Add feature - 'proportion_imports'
    train_data = adding_new_feature(train_data)
    test_data = adding_new_feature(test_data) 

    # Categorical Data Handling
    train_data, test_data = handle_categorial_features(train_data, test_data)
    
    # Outliers Removal
    train_data, train_labels, test_data = handle_outliers(train_data, train_labels, test_data)

    # Normalization
    train_data, test_data = normalize_data(train_data, test_data)

    # Feature Reduction
    train_data = remove_features(train_data)
    test_data = remove_features(test_data)

    # PCA
    # train_data, test_data = perform_pca(train_data, test_data, 0.99) 
    
    return train_data, train_labels, test_data


# train_data_ppc1 = train_data_ppc.copy()
# train_lables_ppc1 = train_lables_ppc.copy()

# test_data_ppc1 = test_data_ppc.copy()
# test_labels_ppc1 = test_labels_ppc.copy()

# preprocessed_train_data, preprocessed_train_labels, preprocessed_test_data = preprocess_data_ALL(train_data_ppc1, train_lables_ppc1, test_data_ppc1)
# preprocessed_test_labels = test_labels_ppc1.copy()


def run_pipeline(train_csv, test_csv, predict_csv):

    train = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_labels = train['label']
    train_data = train.drop('label', axis=1)

    sha256 = test_data['sha256']

    train_data, train_labels, test_data = preprocess_data_ALL(train_data, train_labels, test_data)

    rf_best_model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
    rf_best_model.fit(train_data, train_labels)

    # Predict on the test set using the best model
    probabilities = rf_best_model.predict_proba(test_data)[:, 1]

    test_data = pd.concat([sha256, test_data], axis=1)

    # Create a DataFrame with the predictions
    results_df = pd.DataFrame({'sha256': test_data['sha256'], 'predict_proba': probabilities})

    # Save the results to a CSV file
    results_df.to_csv(predict_csv, index=False)

run_pipeline('TRAIN.csv', 'TEST.csv', 'results.csv')