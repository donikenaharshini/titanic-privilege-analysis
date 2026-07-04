import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_analyzed.csv")
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Chart 1: Survival by class and sex
class_sex = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack() * 100
class_sex.plot(kind="bar", ax=axes[0, 0], color=["#e74c3c", "#3498db"])
axes[0, 0].set_title("Survival Rate (%) by Class and Sex")
axes[0, 0].set_ylabel("Survival Rate (%)")
axes[0, 0].set_xlabel("Passenger Class")
axes[0, 0].legend(title="Sex")
axes[0, 0].tick_params(axis='x', rotation=0)

# Chart 2: Survival by privilege bracket
priv_order = ["Low", "Medium", "High", "Very High"]
priv_data = df.groupby("PrivilegeBracket", observed=True)["Survived"].mean().reindex(priv_order) * 100
priv_data.plot(kind="bar", ax=axes[0, 1], color="#2ecc71")
axes[0, 1].set_title("Survival Rate (%) by Privilege Score Bracket")
axes[0, 1].set_ylabel("Survival Rate (%)")
axes[0, 1].tick_params(axis='x', rotation=15)

# Chart 3: Fare distribution - survivors vs not
sns.boxplot(data=df, x="Survived", y="Fare", ax=axes[1, 0])
axes[1, 0].set_title("Fare Paid: Survivors vs Non-Survivors")
axes[1, 0].set_xticklabels(["Did Not Survive", "Survived"])
axes[1, 0].set_ylim(0, 300)  # zoom in, ignore extreme outliers for clarity

# Chart 4: Survival by family size
family_data = df.groupby("FamilySize")["Survived"].mean() * 100
family_data.plot(kind="bar", ax=axes[1, 1], color="#9b59b6")
axes[1, 1].set_title("Survival Rate (%) by Family Size")
axes[1, 1].set_xlabel("Family Size (incl. self)")
axes[1, 1].set_ylabel("Survival Rate (%)")
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig("titanic_privilege_analysis.png", dpi=150)
print("Saved chart as titanic_privilege_analysis.png")
