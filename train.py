import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("loan_data.csv")

print("Original Data Types:\n")
print(df.dtypes)

# Drop Loan_ID if exists
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Convert ALL non-numeric columns to numeric
le = LabelEncoder()

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Encoding column: {col}")
        df[col] = le.fit_transform(df[col].astype(str))

print("\nAfter Encoding:\n")
print(df.dtypes)

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("\nChecking X sample:\n")
print(X.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved successfully!")