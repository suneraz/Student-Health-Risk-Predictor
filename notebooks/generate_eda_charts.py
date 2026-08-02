import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("eda_plots", exist_ok=True)

df = pd.read_csv("train.csv")

numeric_cols = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
                "step_count", "exercise_duration", "water_intake"]
cat_cols = ["diet_type", "stress_level", "sleep_quality",
            "physical_activity_level", "smoking_alcohol", "gender"]

sns.set_style("whitegrid")

# 1. dataset overview
print("shape:", df.shape)
print()
print(df.head())
print()
df.info()

# 2. summary stats for numeric columns
print()
print(df[numeric_cols].describe())

# 3. duplicate rows check
print()
print("duplicate rows:", df.duplicated().sum())

# 4. unique values in categorical columns
print()
for col in cat_cols:
    print(col, "-", df[col].nunique(), "unique values")
    print(df[col].value_counts())
    print()

# 5. target distribution
fig, ax = plt.subplots(figsize=(6, 4))
df["health_condition"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"], ax=ax)
ax.set_title("Target Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_plots/target_distribution.png")
plt.close()

# 6. missing values
fig, ax = plt.subplots(figsize=(8, 5))
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
(missing / len(df) * 100).plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Missing Values by Column (%)")
ax.set_xlabel("% Missing")
plt.tight_layout()
plt.savefig("eda_plots/missing_values.png")
plt.close()

# 7. numeric features vs target
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x="health_condition", y=col, ax=axes[i], order=["fit", "at-risk", "unhealthy"])
    axes[i].set_title(col)
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("eda_plots/numeric_by_target.png")
plt.close()

# 8. correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Numeric Feature Correlation")
plt.tight_layout()
plt.savefig("eda_plots/correlation.png")
plt.close()

# 9. categorical features vs target
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df["health_condition"], normalize="index")
    ct = ct[["fit", "at-risk", "unhealthy"]]
    ct.plot(kind="bar", stacked=True, ax=axes[i], color=["#55A868", "#4C72B0", "#DD8452"], legend=(i == 0))
    axes[i].set_title(col)
    axes[i].set_ylabel("Proportion")
plt.tight_layout()
plt.savefig("eda_plots/categorical_by_target.png")
plt.close()

print()
print("done, charts saved in eda_plots folder")
