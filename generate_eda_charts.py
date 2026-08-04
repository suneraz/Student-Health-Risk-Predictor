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

# 3.1 dataset overview
print("shape:", df.shape)
print()
print(df.head())
print()
df.info()

# 3.2 missing data
print()
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("missing values per column:")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# 3.3 duplicate rows
print()
print("duplicate rows:", df.duplicated().sum())

# 3.4 class distribution -> Figure 1
print()
print(df["health_condition"].value_counts())
print(df["health_condition"].value_counts(normalize=True).round(3))

fig, ax = plt.subplots(figsize=(6, 4))
df["health_condition"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452", "#55A868"], ax=ax)
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_plots/class_distribution.png")
plt.close()

# 3.5 numerical feature distributions (standalone, not split by class) -> Figure 2
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#4C72B0")
    axes[i].set_title(col)
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("eda_plots/numeric_distributions.png")
plt.close()

# 3.6 numeric feature and target relationships -> Figure 3
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

# 3.7 correlation between numeric features -> Figure 4
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Between Numeric Features")
plt.tight_layout()
plt.savefig("eda_plots/correlation.png")
plt.close()

# 3.8 categorical feature distributions (standalone, not split by class) -> Figure 5
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind="bar", ax=axes[i], color="#55A868")
    axes[i].set_title(col)
    axes[i].set_ylabel("Count")
plt.tight_layout()
plt.savefig("eda_plots/categorical_distributions.png")
plt.close()

# 3.9 categorical feature and target relationships -> Figure 6
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
