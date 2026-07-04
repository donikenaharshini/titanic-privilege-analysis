# Titanic: Did Privilege Outweigh "Women and Children First"?

A data analysis project exploring the Titanic dataset from a social/economic
angle — instead of building a survival prediction model (the common approach),
this project investigates **how much class, wealth, and family size influenced
survival compared to the "women and children first" rule.**

## Project Goal

Most beginner Titanic projects ask: *"Can we predict who survived?"*
This project asks a different question: ***"Was the evacuation really fair,
or did privilege matter more than we think?"***

## Files

| File | Purpose |
|---|---|
| `Titanic-Dataset.csv` | Raw dataset (891 passengers) |
| `step1_clean.py` | Cleans missing values (Age, Embarked, Cabin), engineers `FamilySize` and `HasCabin` |
| `step2_analysis.py` | Core analysis — survival by sex, age, class, fare, family size, and a custom "Privilege Score" |
| `step3_visualize.py` | Generates a 4-panel chart summarizing the findings |
| `titanic_analyzed.csv` | Final cleaned + feature-engineered dataset |
| `titanic_privilege_analysis.png` | Output visualization |

## How to Run

```bash
pip install pandas matplotlib seaborn
python step1_clean.py
python step2_analysis.py
python step3_visualize.py
```

Each script builds on the previous one's output, so run them in order.

## Key Findings

- **Women survived at 74%, men at only 19%** — the "women first" principle largely held.
- **But class had a major effect too**: 1st-class women survived at 97%, while
  3rd-class women survived at only 50% — half the rate.
- **1st-class men (37% survival) survived at nearly the same rate as 3rd-class
  women (50%)** — suggesting wealth/status could rival gender as a survival factor.
- **Family size mattered**: small families (2-4 people) survived better than solo
  travelers or very large families (5+), who were harder to evacuate together.
- A custom **Privilege Score** (combining class, sex, fare, and cabin status)
  shows survival rising steadily from **12% (low privilege)** to **70% (very high privilege)**.

## Methodology Notes

- Missing `Age` values filled with median age (simple imputation for beginner-level analysis).
- Missing `Embarked` values filled with the most frequent port.
- `Cabin` had too many missing values (77%) to impute meaningfully, so it was
  converted into a binary `HasCabin` flag instead.
- `FamilySize` = SibSp + Parch + 1 (self).
- `PrivilegeScore` is a custom weighted score:
  `(4 - Pclass) * 2 + (is_female) * 3 + HasCabin * 2 + (above_median_fare) * 2`

## Tools Used

Python, pandas, matplotlib, seaborn

## Dataset Source

[Titanic dataset](https://www.kaggle.com/c/titanic/data) — a well-known public
dataset of Titanic passengers and survival outcomes.
