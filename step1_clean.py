import pandas as pd

# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Quick look at the data
print("Shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# --- Cleaning ---
# Age has missing values -> fill with median age (simple, beginner-friendly approach)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked has a couple missing -> fill with the most common port
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has too many missing values to fill meaningfully -> just flag whether cabin is known
df["HasCabin"] = df["Cabin"].notnull().astype(int)

# Create FamilySize = siblings/spouses + parents/children + self
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Save cleaned version
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned data saved as titanic_cleaned.csv")
print(df.head())
