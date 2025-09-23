# regression_autotrain.py
from pycaret.regression import *        # Regression task
from pycaret.regression import save_model, load_model

import pandas as pd

# -----------------------
# Load your CSV
# -----------------------
data = pd.read_csv('Superstore.csv')            # <- replace with your CSV file
# Optional: data = pd.get_dummies(data, drop_first=True)

# -----------------------
# Step 1: Setup (Regression)
# -----------------------
regression_setup = setup(
    data=data,
    target='Sales',               # <- replace with your numeric target column
    train_size=0.8,
    session_id=123,
    normalize=True,
    transformation=True,
    remove_outliers=True,
    outliers_threshold=0.05,
    feature_selection=True,
    polynomial_features=False,
    # silent=True    # uncomment to suppress prompts (useful in scripts)
)

# -----------------------
# Step 2: Compare Models
# -----------------------
# Efficient approach: get top 5 models and use the first as the 'best'
top_models = compare_models(n_select=5, sort='R2')   # returns a list of trained models (top 5 by R2)
best_model = top_models[0]                            # best (highest R2)

# If you prefer only the single best model (and don't need top_n list), use:
# best_model = compare_models(sort='R2')

# -----------------------
# Step 3: Leaderboard (DataFrame) and save to CSV
# -----------------------
leaderboard = pull()                 # DataFrame of results from last compare_models() call
print(leaderboard)
leaderboard.to_csv("leaderboard_regression_results.csv", index=False)

# Optional: display a subset of columns
print(leaderboard[['Model', 'MAE', 'RMSE', 'R2']])

# -----------------------
# Step 4: Save models to disk
# -----------------------
# Save the best model
save_model(best_model, 'best_regression_model')

# Save the top N models returned earlier
for i, model in enumerate(top_models, start=1):
    save_model(model, f'regression_model_top{i}')

# Example of loading a saved model later:
# loaded = load_model('best_regression_model')

# -----------------------
# Notes / Tips
# -----------------------
# - compare_models(n_select=k) returns a list of k models (already trained).
# - If you call compare_models() again later, PyCaret may retrain models; use the returned list to avoid re-running.
# - Use `models()` to see available regression model short-codes if you want to include/exclude specific models.
#     print(models())
#
# - To explicitly train only a few models:
#     top = compare_models(include=['lr','rf','lightgbm'], sort='R2')
#
# - For non-interactive scripts, you can pass silent=True to setup() to avoid prompts:
#     setup(..., silent=True)

# Specifications, evaluation, 