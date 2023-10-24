import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

epochs = 20
batch_size = 2048
start_learning_rate = 0.0001
k_folds = 5  # specify the number of folds
kf = KFold(n_splits=k_folds)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

decrease_interval = 2
lr_decay = 0.9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features.iloc[idx].values, dtype=torch.float), torch.tensor(self.labels.iloc[idx], dtype=torch.float)


# 1. Load data from data/train.csv and data/test.csv
# 2. Preprocess data
# 3. Create model via PyTorch
# 4. Train model
# 5. Evaluate model
# 6. Save model
# 7. Make prediction
# 8. Save prediction to data/prediction.csv in the same format as data/sample_submission.csv

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

    # Calculate ratio of lines of code to essential complexity
    df["ratio_loc_essential"] = df["total_loc"] / df["ev(g)"]

    # Calculate ratio of lines of code to maintainability index
    df["ratio_loc_maintainability"] = df["total_loc"] / df["b"]

    # Calculate ratio of lines of code to McCabe's cyclomatic complexity
    df["ratio_loc_mccabe_cyclo"] = df["total_loc"] / df["lOCode"]

    # Calculate ratio of lines of code to McCabe's essential complexity
    df["ratio_loc_mccabe_essential"] = df["total_loc"] / df["lOComment"]

    # Calculate ratio of lines of code to McCabe's maintainability index
    df["ratio_loc_mccabe_maintainability"] = df["total_loc"] / df["lOBlank"]

    # Calcuulate ratio of lines of code to Halstead difficulty
    df["ratio_loc_halstead_difficulty"] = df["total_loc"] / df["d"]    

    if not test:
        # Find most correlated features with defects
        corr = df.corr()
        corr_defects = corr["defects"].abs().sort_values(ascending=False)
        correlated_columns.extend(corr_defects[1:11].index.tolist())

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

test_ids = test["id"]
print("Preprocessing test data...")
test, _ = preprocess(test, test=True)

data = X_train.values
labels = y_train.values

# 3. Create model via PyTorch using the shape of the data as input
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 8)
        self.fc6 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.sigmoid(self.fc6(x))
    
total_loss = 0.0
for fold, (train_indices, val_indices) in enumerate(kf.split(data)):
    learning_rate = start_learning_rate
    print(f"Starting Fold {fold + 1}/{k_folds}...")
    X_train_fold, X_val_fold = data[train_indices], data[val_indices]
    y_train_fold, y_val_fold = labels[train_indices], labels[val_indices]

    train_dataset = CustomDataset(pd.DataFrame(X_train_fold), pd.Series(y_train_fold))
    val_dataset = CustomDataset(pd.DataFrame(X_val_fold), pd.Series(y_val_fold))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print("Creating model...")
    net = Net(X_train_fold.shape[1]).to(device)

    # 4. Train model
    print("Training model...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        if (epoch + 1) % decrease_interval == 0:
            learning_rate = learning_rate * lr_decay
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        print(f"Starting epoch {epoch + 1} with learning rate {learning_rate}...")
        running_loss = 0.0
        for inputs, labels_set in train_loader:
            inputs, labels_set = inputs.to(device), labels_set.to(device).unsqueeze(1) 

            optimizer.zero_grad()        
            outputs = net(inputs)
            loss = criterion(outputs, labels_set)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(inputs)  # Multiply by batch size to account for the mean loss
            inputs, labels_set = None, None  # Free memory

        print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader.dataset):.3f}")


    # 5. Evaluate model
    print("Evaluating model...")
    net.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels_set in val_loader:
            inputs, labels_set = inputs.to(device), labels_set.to(device).unsqueeze(1)

            outputs = net(inputs)
            loss = criterion(outputs, labels_set)

            running_loss += loss.item() * len(inputs)

            inputs, labels_set = None, None  # Free memory

        print(f"Validation Loss: {running_loss / len(val_loader.dataset):.3f}")
        total_loss += running_loss / len(val_loader.dataset)
        
    net = None  # Free memory

print(f"Average Validation Loss: {total_loss / k_folds:.3f}")

train_dataset = CustomDataset(pd.DataFrame(data), pd.Series(labels))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Creating model...")
net = Net(data.shape[1]).to(device)

learning_rate = start_learning_rate

# 4. Train model
print("Training model...")
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(epochs):
    if (epoch + 1) % decrease_interval == 0:
        learning_rate = learning_rate * lr_decay
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print(f"Starting epoch {epoch + 1} with learning rate {learning_rate}...")
    running_loss = 0.0
    for inputs, labels_set in train_loader:
        inputs, labels_set = inputs.to(device), labels_set.to(device).unsqueeze(1) 

        optimizer.zero_grad()        
        outputs = net(inputs)
        loss = criterion(outputs, labels_set)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(inputs)  # Multiply by batch size to account for the mean loss
        inputs, labels_set = None, None  # Free memory

    print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader.dataset):.3f}")

# 6. Save model
print("Saving model...")
torch.save(net.state_dict(), "model.pt")

# 7. Make prediction

test_dataset = CustomDataset(test, pd.DataFrame(np.zeros(len(test))))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Making prediction...")
net.eval()
with torch.no_grad():
    predictions = []
    for inputs, _ in test_loader:
        inputs = inputs.to(device)

        outputs = net(inputs)
        predictions.extend(outputs.squeeze(1).tolist())

# 8. Save prediction to data/prediction.csv in the same format as data/sample_submission.csv
print("Saving prediction...")
submission = pd.DataFrame({"id": test_ids, "defects": predictions})
submission.to_csv("data/prediction.csv", index=False)