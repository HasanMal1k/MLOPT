from pycaret.classification import *
from pycaret.classification import save_model, load_model
# What we import decides what task we are performing. We never expliity specify 
# specify what ML task to perform. We simply import the relevant module and it imports functions
# relevant to that task e.g. setup()

# Load your CSV file
import pandas as pd
data = pd.read_csv('Iris.csv')
#data = pd.get_dummies(data, drop_first=True)


# ✅ Step 1: Setup PyCaret (Mandatory Step)
#data.columns = data.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]+', '', regex=True)

classification_setup = setup(
    data=data,                # your pandas DataFrame
    target='Species',   # replace with your actual target column name
    train_size=0.8,           # 80% training, 20% test
    session_id=123,           # for reproducibility
    normalize=True,           # optional: scales numerical features
   # preprocess=True,
    transformation=True,     # optional: applies power transforms like Box-Cox
    remove_outliers=True,     # remove outliers from training set
    outliers_threshold=0.05,  # % of outliers to remove
    feature_selection=True,   # automatically select top features5
   # feature_interaction=True, # generate new interaction features
    polynomial_features=False,# optionally add polynomial features
   # ignore_low_variance=True, # remove near-zero variance features
   # silent=True               # avoid asking for confirmation
)

# ✅ Step 2: Compare Models (Auto-Trains All Suitable Models)
best_model = compare_models(sort='Accuracy')  # You can also sort by 'MAE', 'RMSE', etc.

# ✅ Step 3: Display Leaderboard (Full Table)
leaderboard = pull()
print(leaderboard)



# Comparison Options
#compare_models(sort='MAE')     # Mean Absolute Error
#compare_models(sort='RMSE')    # Root Mean Square Error
#compare_models(sort='R2')      # R-squared (default)


# Leaderboard options
leaderboard = pull()
#leaderboard[['Model', 'MAE', 'RMSE', 'R2']]
leaderboard.to_csv("leaderboard_results.csv", index=False)


leaderboard[['Model', 'Accuracy']]



#########################################################################################3
##########################################################################################
##########################################################################################

#                        Code to access the auto-trained models

##########################################################################################
##########################################################################################
##########################################################################################



# Save the best model
save_model(best_model, 'best_model')

# Save top 3 models
# for i, model in enumerate(top3, start=1):
#     save_model(model, f'model_{i}')
