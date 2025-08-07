from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('CVD_Dataset.csv')
print(f"Dataset loaded with shape: {df.shape}")

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, error_score='raise')

# Define X_train and y_train
print("Preparing features and target...")
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Convert categorical variables to numerical variables
print("Processing categorical variables...")
label_encoder = LabelEncoder()
X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])
X_test['Sex'] = label_encoder.transform(X_test['Sex'])

# Convert 'Blood Pressure (mmHg)' column to float - handle missing values
def process_blood_pressure(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str) and '/' in x:
        try:
            return float(x.split('/')[0])
        except:
            return np.nan
    try:
        return float(x)
    except:
        return np.nan

print("Processing blood pressure data...")
X_train['Blood Pressure (mmHg)'] = X_train['Blood Pressure (mmHg)'].apply(process_blood_pressure)
X_test['Blood Pressure (mmHg)'] = X_test['Blood Pressure (mmHg)'].apply(process_blood_pressure)

# Convert 'Smoking Status' column to numerical values
def process_smoking_status(x):
    if pd.isna(x):
        return np.nan
    if x == 'Y':
        return 1
    elif x == 'N':
        return 0
    else:
        return np.nan

print("Processing smoking status...")
X_train['Smoking Status'] = X_train['Smoking Status'].apply(process_smoking_status)
X_test['Smoking Status'] = X_test['Smoking Status'].apply(process_smoking_status)

# Handle missing values by imputing with median for numerical columns
print("Handling missing values...")

# Create imputer for numerical columns
numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Get numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=[np.number]).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

print(f"Numerical columns: {list(numerical_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

# Impute missing values
if len(numerical_cols) > 0:
    X_train[numerical_cols] = numerical_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = numerical_imputer.transform(X_test[numerical_cols])

if len(categorical_cols) > 0:
    X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

# Convert remaining categorical columns to numerical
print("Converting remaining categorical columns...")
for col in categorical_cols:
    if col != 'Sex':  # Already processed
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Ensure all data is numeric
X_train = X_train.astype(float)
X_test = X_test.astype(float)

print(f"Final training set shape: {X_train.shape}")
print(f"Missing values in X_train: {X_train.isnull().sum().sum()}")
print(f"Missing values in y_train: {y_train.isnull().sum()}")

# Fit the grid search to the data
print("Starting grid search...")
try:
    grid_search.fit(X_train, y_train)
    print("Grid search completed successfully!")
    
    # Print the best parameters
    print("Best parameters:", grid_search.best_params_)
    
    # Print the best score
    print("Best score:", grid_search.best_score_)
    
    # Make predictions on test set
    y_pred = grid_search.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
except ValueError as e:
    print("An error occurred while fitting the grid search to the data:", e)
    print("Data shape:", X_train.shape)
    print("Missing values in X_train:", X_train.isnull().sum().sum())
    print("Missing values in y_train:", y_train.isnull().sum())
    print("Data types in X_train:")
    print(X_train.dtypes)
