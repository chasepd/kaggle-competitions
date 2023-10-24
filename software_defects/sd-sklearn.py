from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

k_folds = 5  # specify the number of folds
kf = KFold(n_splits=k_folds)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)

print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

scaler = StandardScaler()

print("Loading data...")
# 1. Load data from data/train.csv and data/test.csv
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

scaler = StandardScaler()

correlated_columns = []

# 2. Preprocess data
def preprocess(df, test=False):
    df = df.drop(columns=["id"])

    # Feature engineering
    
    # Combine loc and IOCode into a new feature that is the average of the two, and then drop the two features
    df["total_loc"] = (df["loc"] + df["lOCode"]) / 2
    
    # Calculate ratio of comments to lines of code
    df["ratio_comments_loc"] = df["lOComment"] / df["total_loc"] 

    #Calculate ratio of lines of code to branch count
    df["ratio_loc_branch"] = df["total_loc"] / df["branchCount"]

    #Calculate ratio of lines of code to cyclomatic complexity
    df["ratio_loc_cyclo"] = df["total_loc"] / df["v(g)"]

    # Calculate ratio of cyclomatic complexity to essential complexity
    df["ratio_cyclo_essential"] = df["v(g)"] / df["ev(g)"]

    # Calcuulate ratio of lines of code to Halstead difficulty
    df["ratio_loc_halstead_difficulty"] = df["total_loc"] / df["d"]    

    if not test:
        # Find 10 most correlated features with defects
        corr = df.corr()
        corr_defects = corr["defects"].abs().sort_values(ascending=False)
        correlated_columns.extend(corr_defects[1:10].index.tolist())

    uncorrelated_columns =[]
    uncorrelated_columns.extend(df.columns.tolist())
    if not test:
        uncorrelated_columns.remove("defects")
    uncorrelated_columns = [x for x in uncorrelated_columns if x not in correlated_columns]
    
    # Drop columns that are not correlated with defects
    df = df.drop(columns=uncorrelated_columns)
    
    # Fill missing values with 0
    df = df.fillna(-1)
    df = df.replace([np.inf, -np.inf], -1)

    # Scale all features    
    # If not test data, do a fit_transform on the training data
    if not test:
        X_train = df.drop(columns=["defects"])
        y_train = df["defects"]
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        return X_train, y_train
    # If test data, do a transform on the test data
    else:
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)    
        return df, None



print("Preprocessing training data...")
X_train, y_train = preprocess(train)

#Change y_train to 1 if y_train > 0 using pandas
y_train = y_train.apply(lambda x: 1 if x > 0 else 0)
print(y_train)

test_ids = test["id"]
print("Preprocessing test data...")
test, _ = preprocess(test, test=True)

data = X_train.values
labels = y_train.values

total_loss = 0.0
for fold, (train_indices, val_indices) in enumerate(kf.split(data)):
    print(f"Starting Fold {fold + 1}/{k_folds}...")
    X_train_fold, X_val_fold = data[train_indices], data[val_indices]
    y_train_fold, y_val_fold = labels[train_indices], labels[val_indices]

    print("Creating RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=10, random_state=seed)

    print("Training model...")
    model.fit(X_train_fold, y_train_fold)

    print("Evaluating model...")
    val_score = model.score(X_val_fold, y_val_fold)
    print(f"Validation Score: {val_score}")

    total_loss += val_score

print(f"Average Validation Score: {total_loss / k_folds}")

print("Training final model...")
final_model = RandomForestRegressor(n_estimators=1000, random_state=seed)
final_model.fit(data, labels)

print("Making prediction...")
predictions = final_model.predict(test.values)

print("Saving prediction...")
submission = pd.DataFrame({"id": test_ids, "defects": predictions})
submission.to_csv("data/prediction-sklearn.csv", index=False)