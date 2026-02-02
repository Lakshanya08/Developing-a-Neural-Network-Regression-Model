# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model

<img width="1137" height="817" alt="Screenshot 2026-02-02 094459" src="https://github.com/user-attachments/assets/fbbc1aec-3485-44b1-8e6d-d6d1901caf26" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Lakshanya.N

### Register Number:212224230136

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/exp1 - Sheet1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#Name:Lakshanya.N
#Register Number:212224230136
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1=nn.Linear(1,8) #fc=fully connected,nn=neural network,1.input
        self.fc2=nn.Linear(8,10)#2.hidden layer
        self.fc3=nn.Linear(10,1)#3.output
        self.relu=nn.ReLU() #activation part,Rectified Linear Unit
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()#class name
criterion=nn.MSELoss()#Mean square
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)#learning rate,Root means quare

# Name:Lakshanya.N
# Register Number: 212224230136
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="237" height="444" alt="image" src="https://github.com/user-attachments/assets/b3a0885a-571d-4e59-a07e-720adef821c2" />

### OUTPUT:
### Input and Output:
<img width="295" height="252" alt="image" src="https://github.com/user-attachments/assets/450707c8-f56d-4216-b94c-52ffde285a36" />

### Epoch and Loss:
<img width="522" height="232" alt="image" src="https://github.com/user-attachments/assets/a4e2d227-b0f9-4b1c-8fc2-0e108db0ac9c" />

### Test Loss:
<img width="441" height="41" alt="image" src="https://github.com/user-attachments/assets/cdd50486-e4d1-4a68-ac2c-f9f34b322523" />

### Training Loss Vs Iteration Plot
<img width="855" height="589" alt="image" src="https://github.com/user-attachments/assets/b4ebdd7b-73f7-4edb-8ba5-d0f49711aba8" />


### New Sample Data Prediction
<img width="544" height="37" alt="image" src="https://github.com/user-attachments/assets/2ddd25dd-f5bd-437a-af27-c78db8e58922" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
