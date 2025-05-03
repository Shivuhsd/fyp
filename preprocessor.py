# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler

# # Modify the categorical pipeline to handle unknown categories more gracefully
# def create_categorical_pipeline():
#     return Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # handle_unknown='ignore'
#     ])


# def create_numerical_pipeline():
#     """Creates the preprocessing pipeline for numerical features."""
#     return Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])

# def create_preprocessor():
#     """Combines categorical and numerical pipelines into a ColumnTransformer."""
#     # Define categorical and numerical features
#     categorical_features = ['Name of the device', 'Location of event', 'Gender', 'Past history', 'Nature of Event']
#     numerical_features = ['Age of the patient']

#     # Create pipelines
#     categorical_pipeline = create_categorical_pipeline()
#     numerical_pipeline = create_numerical_pipeline()

#     # Combine pipelines into a single ColumnTransformer
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numerical_pipeline, numerical_features),
#             ('cat', categorical_pipeline, categorical_features)
#         ]
#     )

#     return preprocessor

# def fit_preprocessor(preprocessor, data):
#     """Fits the preprocessor to the data."""
#     # Check if all expected columns are present
#     expected_columns = ['Name of the device', 'Location of event', 'Gender', 'Past history', 'Nature of Event', 'Age of the patient']
#     for column in expected_columns:
#         if column not in data.columns:
#             raise ValueError(f"Expected column '{column}' not found in data.")

#     # Fit the preprocessor
#     preprocessor.fit(data)
#     return preprocessor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def create_categorical_pipeline():
    """Creates the preprocessing pipeline for categorical features."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # Removed sparse=False
    ])

def create_numerical_pipeline():
    """Creates the preprocessing pipeline for numerical features."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def create_preprocessor():
    """Combines categorical and numerical pipelines into a ColumnTransformer."""
    categorical_features = ['Name of the device', 'Location of event', 'Gender', 'Past history', 'Nature of Event']
    numerical_features = ['Age of the patient']

    categorical_pipeline = create_categorical_pipeline()
    numerical_pipeline = create_numerical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'  # Keep any other columns as is (useful for matching features)
    )

    return preprocessor

def fit_preprocessor(preprocessor, data):
    """Fits the preprocessor to the data."""
    expected_columns = ['Name of the device', 'Location of event', 'Gender', 'Past history', 'Nature of Event', 'Age of the patient']
    for column in expected_columns:
        if column not in data.columns:
            raise ValueError(f"Expected column '{column}' not found in data.")

    preprocessor.fit(data)
    return preprocessor

def main():
    # Load the data
    data = pd.read_excel('data/Updated_TEST_600_modified.xlsx')
    
    # Drop unnecessary columns
    data = data.drop(columns=['Manufacturer name'])

    # Fill missing values
    data['Past history'] = data['Past history'].fillna('No Past History')

    # Create and fit the preprocessor
    preprocessor = create_preprocessor()
    fitted_preprocessor = fit_preprocessor(preprocessor, data)

    # Save the preprocessor
    joblib.dump(fitted_preprocessor, 'models/preprocessor.joblib')

if __name__ == '__main__':
    main()
