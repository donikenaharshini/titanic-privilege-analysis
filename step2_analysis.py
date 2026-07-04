import pandas as pd

df = pd.read_csv("titanic_cleaned.csv")

# ---------------------------------------------------
# QUESTION 1: Did "women and children first" actually hold?
# ---------------------------------------------------
print("=== Survival Rate by Sex ===")
print(df.groupby("Sex")["Survived"].mean().round(3) * 100)

print("\n=== Survival Rate by Age Group ===")
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 40, 60, 100],
                         labels=["Child (0-12)", "Teen (13-18)", "Adult (19-40)",
                                 "Middle Age (41-60)", "Senior (60+)"])
print(df.groupby("AgeGroup")["Survived"].mean().round(3) * 100)

# ---------------------------------------------------
# QUESTION 2: Did class/wealth matter more than gender/age?
# ---------------------------------------------------
print("\n=== Survival Rate by Passenger Class ===")
print(df.groupby("Pclass")["Survived"].mean().round(3) * 100)

print("\n=== Survival Rate by Class AND Sex (combined) ===")
print(df.groupby(["Pclass", "Sex"])["Survived"].mean().round(3) * 100)

print("\n=== Average Fare Paid: Survivors vs Non-Survivors ===")
print(df.groupby("Survived")["Fare"].mean().round(2))

# ---------------------------------------------------
# QUESTION 3: Did family size help or hurt?
# ---------------------------------------------------
print("\n=== Survival Rate by Family Size ===")
print(df.groupby("FamilySize")["Survived"].mean().round(3) * 100)

# ---------------------------------------------------
# QUESTION 4: Build a simple "Privilege Score"
# Higher score = more "privileged" (higher class, female, cabin known, higher fare)
# ---------------------------------------------------
df["PrivilegeScore"] = (
    (4 - df["Pclass"]) * 2 +          # 1st class = 6 pts, 2nd = 4, 3rd = 2
    (df["Sex"] == "female").astype(int) * 3 +  # being female = +3
    df["HasCabin"] * 2 +              # having a known cabin = +2
    (df["Fare"] > df["Fare"].median()).astype(int) * 2  # paying above-median fare = +2
)

print("\n=== Survival Rate by Privilege Score Bracket ===")
df["PrivilegeBracket"] = pd.cut(df["PrivilegeScore"], bins=[-1, 3, 6, 9, 13],
                                  labels=["Low", "Medium", "High", "Very High"])
print(df.groupby("PrivilegeBracket")["Survived"].mean().round(3) * 100)

df.to_csv("titanic_analyzed.csv", index=False)
print("\nAnalyzed data saved as titanic_analyzed.csv")
