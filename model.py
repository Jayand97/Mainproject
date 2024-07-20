import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import bz2
import pickle
import warnings
#warnings.filterwarnings('ignore')
import joblib


# Load the  dataset
data = pd.read_csv('fraudtransaction.csv')
print("Loaded")

# Drop unwanted columns which may affect data and result
data = data.drop(columns=['isFlaggedFraud','Unnamed: 0'])
print("droped")

# Null values are at numeric columns and are skewed so,fill null values with median
imputer = SimpleImputer(strategy='median')
data[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = imputer.fit_transform(data[['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])
    
print("filled")

# Handling Outlier using IQR ,since the data is right skewed
columns_C = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
def handle_outliers(data, columns_C):
    for col in columns_C:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[f'{col}_cleaned'] = np.where(data[col] > upper_bound, upper_bound,
                                       np.where(data[col] < lower_bound, lower_bound, data[col]))
    return data

# Handle outliers in each column of numeric columns
data = handle_outliers(data, columns_C)
#print("Outlier handled")

# Perform label encoding on the 'type' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# View the mapping of labels to integers
print(le.classes_)
print(le.transform(le.classes_))
print("Label encoded")

# Prepare features and target variable
features = ['type', 'amount_cleaned', 'oldbalanceOrg_cleaned', 'newbalanceOrig_cleaned']
X = data[features]
y = data['isFraud']


# Apply Standard Scalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = y  # No scaling needed for y, as it's a binary target variable
print("Scaled")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled,y_scaled)
print("Smote")



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Train test")

# Create a Random Forest model with the best parameters
rf_model = RandomForestClassifier(n_estimators=200,
                                  min_samples_split=2,
                                  min_samples_leaf=4,
                                  max_depth=20,
                                  random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)
print("Trained")

# Evaluate the model on the test data
y_pred = rf_model.predict(X_test)
print("Model accuracy:", rf_model.score(X_test, y_test))

# Save the model and scaler to a pickle file

#pickle.dump(rf_model, open('model.pickle', 'wb'))


#pickle.dump(scaler, open('scaler.pickle', 'wb'))

# Save the model with compression
with bz2.open('model.pbz2', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the scaler with compression
with bz2.open('scaler.pbz2', 'wb') as f:
    pickle.dump(scaler, f)

