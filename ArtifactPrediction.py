import sqlite3
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. LOAD THE DATA FROM SQLITE DATABASES
# -----------------------------------------------------------------------------

# Path to your local sqlite files (update these to your actual paths)
TRAIN_DB_PATH = "/Users/justinhenschen/Desktop/TrainingData.sqlite"
TEST_DB_PATH = "/Users/justinhenschen/Desktop/TestingData.sqlite"

# Table name within the SQLite databases
# (adjust if your DB uses a different table name).
TABLE_NAME = "ExportedEvents"

# Read the training data from the TrainingData.sqlite
conn_train = sqlite3.connect(TRAIN_DB_PATH)
df_train = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn_train)
conn_train.close()

# Read the testing data from the TestingData.sqlite
conn_test = sqlite3.connect(TEST_DB_PATH)
df_test = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn_test)
conn_test.close()

# -----------------------------------------------------------------------------
# 2. PREPARE FEATURES AND LABEL
# -----------------------------------------------------------------------------
# We assume the columns are:
#    Name (str), SubEvent (str), FilePath (str), Flags (str), Signature (str), OfInterest (int or bool)
# Our goal is to predict OfInterest based on the other columns.
#
# If OfInterest is stored as text (e.g., 'Yes'/'No'), you can map it to 1/0.
# This example assumes it's already numeric (0 or 1). Adjust as needed.

feature_cols = ["Name", "SubEvent", "FilePath", "Flags", "Signature"]
label_col = "OfInterest"

# Drop any rows in training data that do not have a valid label
df_train = df_train.dropna(subset=[label_col])

X_train = df_train[feature_cols]
y_train = df_train[label_col]

X_test = df_test[feature_cols]
# We'll keep the original test DataFrame so we can append predictions next
df_test_original = df_test.copy()

# -----------------------------------------------------------------------------
# 3. BUILD A PREPROCESSING AND CLASSIFICATION PIPELINE
# -----------------------------------------------------------------------------
# We will treat all feature columns as categorical/text. If some columns are numeric,
# you can treat them differently (e.g., scaling). For text columns that are large or
# free-form, consider TF-IDF encoding, etc. For now, let's keep it simple.

categorical_features = feature_cols  # all columns in this example
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# ColumnTransformer allows us to apply transformations selectively to columns.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"  # drop other columns not in feature_cols
)

# Choose a classification model. RandomForest is just one example:
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline that first applies preprocessing, then fits the classifier.
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# -----------------------------------------------------------------------------
# 4. TRAIN THE MODEL
# -----------------------------------------------------------------------------
model_pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 5. MAKE PREDICTIONS ON THE TEST SET
# -----------------------------------------------------------------------------
y_test_pred = model_pipeline.predict(X_test)

# Optionally, you can also get predicted probabilities:
y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]  # Probability of class "1"

# -----------------------------------------------------------------------------
# 6. SAVE THE RESULTS
# -----------------------------------------------------------------------------
# Create new columns in our test DataFrame to store predictions.
df_test_original["Predicted_OfInterest"] = y_test_pred
df_test_original["Prediction_Probability"] = y_test_prob

# You can now save this DataFrame to a CSV or write back to an SQLite DB
output_csv_path = "Predictions.csv"
df_test_original.to_csv(output_csv_path, index=False)

# If you want to create a new table in your TestingData.sqlite:
conn_test = sqlite3.connect(TEST_DB_PATH)
df_test_original.to_sql("TestPredictions", conn_test, if_exists="replace", index=False)
conn_test.close()

print("Done! Predictions are saved to Predictions.csv and TestPredictions table in TestingData.sqlite.")
