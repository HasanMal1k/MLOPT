from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.datasets import load_boston, load_iris
import pandas as pd

# Example 1: Classification (Iris dataset)
iris = load_iris()
X_class, y_class = iris.data, iris.target
mi_class = mutual_info_classif(X_class, y_class)
print("MI scores (classification):", mi_class)

# Example 2: Regression (Boston Housing dataset)
boston = load_boston()
X_reg, y_reg = boston.data, boston.target
mi_reg = mutual_info_regression(X_reg, y_reg)
print("MI scores (regression):", mi_reg)


#################################################

#    Feature selection based MI proportion covering 60% certainty

#################################################

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris

# Example dataset (classification)
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names

# Step 1: Compute MI scores
mi_scores = mutual_info_classif(X, y, random_state=42)

# Put results in a DataFrame for readability
mi_df = pd.DataFrame({
    "Feature": feature_names,
    "MI_Score": mi_scores
}).sort_values(by="MI_Score", ascending=False).reset_index(drop=True)

# Step 2: Calculate cumulative coverage
mi_df["Cumulative"] = mi_df["MI_Score"].cumsum()
mi_df["Cumulative_Percent"] = mi_df["Cumulative"] / mi_df["MI_Score"].sum()

# Step 3: Select features covering 60% of MI
selected_features = mi_df[mi_df["Cumulative_Percent"] <= 0.60]["Feature"].tolist()

print("Mutual Information Scores:")
print(mi_df)
print("\nSelected Features (covering 60% of MI):")
print(selected_features)


########   Hum choose kreinge features based on 60% certainty
######## and leave the rest to the user to choose features as they wish.




