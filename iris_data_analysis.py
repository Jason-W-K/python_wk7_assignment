# Task 1: Load and Explore the Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset with error handling
try:
    iris_raw = load_iris()
    iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Display first few rows
print("\n📌 First 5 rows of the dataset:")
print(iris_df.head())

# Check data types and missing values
print("\n🔍 Dataset Info:")
print(iris_df.info())

print("\n🧼 Missing Values:")
print(iris_df.isnull().sum())

# No missing values in Iris dataset, so no cleaning needed

# Task 2: Basic Data Analysis
print("\n📊 Basic Statistics:")
print(iris_df.describe())

# Group by species and compute mean of numerical columns
grouped_means = iris_df.groupby('species').mean()
print("\n📈 Mean values grouped by species:")
print(grouped_means)

# Observations
print("\n🧠 Observations:")
print("- Setosa has the smallest petal and sepal dimensions.")
print("- Virginica tends to have the largest petal length and width.")
print("- Clear separation in petal dimensions across species, useful for classification.")

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart (simulated time-series using index)
plt.figure(figsize=(10, 5))
plt.plot(iris_df.index, iris_df['sepal length (cm)'], label='Sepal Length')
plt.plot(iris_df.index, iris_df['petal length (cm)'], label='Petal Length')
plt.title("🌿 Sepal vs Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, ci=None)
plt.title("🌸 Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title("📊 Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title("🔬 Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()
