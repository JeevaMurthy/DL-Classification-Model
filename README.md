# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.



## DESIGN STEPS
### STEP 1: 
Load Dataset – Import Iris dataset, extract features (X) and labels (y).

### STEP 2: 
Preprocess Data – Split into train/test sets, standardize features, convert to tensors.


### STEP 3: 
Create DataLoader – Batch training and testing data using TensorDataset and DataLoader.


### STEP 4: 
Define Model – A feedforward neural network with: Input layer → Hidden (16 neurons, ReLU) Hidden (8 neurons, ReLU) → Output (3 classes).


### STEP 5: 
Set Loss & Optimizer – Use CrossEntropyLoss and Adam optimizer.



### STEP 6: 
Train Model – For each epoch: Forward pass → compute predictions. Calculate loss → backward pass. Update weights with optimizer.

### STEP 7:
Evaluate Model – Test on unseen data, compute accuracy, confusion matrix, and classification report.

### STEP 8:
Visualize Results – Plot confusion matrix using seaborn heatmap.

### STEP 9:
Sample Prediction – Predict a test sample and compare predicted vs. actual class.




## PROGRAM

#### Name: Jeeva K

#### Register Number: 212223230090

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load Iris dataset
iris = load_iris()
print(iris)
X = iris.data  # Features
y = iris.target  # Labels (already numerical)

# Convert to DataFrame for easy inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df)

# Display first and last 5 rows
print("First 5 rows of dataset:\n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define Neural Network Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,3)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()
    if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Initialize model, loss function, and optimizer
model =IrisClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluate the model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)


print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# Make a sample prediction
sample_input = X_test[5].unsqueeze(0)  # Removed unnecessary .clone()
with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = iris.target_names[predicted_class_index]

print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')

```

### Dataset Information
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/9eb7308d-ce02-4c42-b0c3-7366f312d54a" />


### OUTPUT

## Confusion Matrix
<img width="240" height="120" alt="image" src="https://github.com/user-attachments/assets/57644609-99e6-4481-b907-13845b5d47f3" />

</br>
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/b22820df-b848-4410-8314-811794bb3f62" />


## Classification Report
<img width="600" height="240" alt="image" src="https://github.com/user-attachments/assets/580e51e5-36ff-424e-a4e5-5305494683ad" />


### New Sample Data Prediction
<img width="400" height="64" alt="image" src="https://github.com/user-attachments/assets/13d966e9-c092-4264-bea8-df43749f8a41" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch.
