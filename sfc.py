#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# In[2]:


# Hyperparameters
sequence_length = 5  # Number of previous timesteps to predict the next
reservoir_size = 50
readout = Ridge(alpha=1.0)
mse_progress = []
new_num = 7


# In[3]:


# Generate data function
def generate_data(time_steps, start):
    np.random.seed(42)
    #time = np.arange(start, time_steps)
    time = np.linspace(start, start + time_steps - 1, num = time_steps)
    temperature_data = 20 + 10 * np.sin(0.02 * time) + np.random.normal(0, 1, time_steps)
    return temperature_data


# In[4]:


# Create input and target pairs for time-series prediction function
def create_pairs(temperature_data):
    input_data = np.array([temperature_data[i:i + sequence_length] for i in range(len(temperature_data) - sequence_length)])
    target_data = temperature_data[sequence_length:]
    return input_data, target_data


# In[5]:


# Define a simple Liquid State Machine
class LiquidStateMachine:
    def __init__(self, reservoir_size, sparsity=0.1):
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        self.reservoir = np.random.rand(reservoir_size, reservoir_size) < sparsity
        self.reservoir = self.reservoir * np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        self.input_weights = np.random.uniform(-1, 1, reservoir_size)
        self.state = np.zeros(reservoir_size)

    def step(self, input_signal):
        # Update the reservoir state
        input_term = self.input_weights * input_signal
        self.state = np.tanh(np.dot(self.reservoir, self.state) + input_term)
        return self.state

    def process_input(self, input_sequence):
        # Process an input sequence through the reservoir
        states = []
        for input_signal in input_sequence:
            state = self.step(input_signal)
            states.append(state)
        return np.array(states)


# In[6]:


# Process data through the LSM
def get_states(input_data):
    states = np.array([lsm.process_input(sequence) for sequence in input_data])
    return states[:, -1, :]  # Take the final state of each sequence


# In[7]:


# Plot the results
def plot_results(test_target, predictions):
    plt.plot(test_target, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('LSM Weather Forecasting')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()


# In[8]:


# Generate data
temperature_data = generate_data(2000, 0)

# Create input and target pairs for time-series prediction
input_data, target_data = create_pairs(temperature_data)

# Split into training and testing sets
train_size = int(0.8 * len(input_data))
train_input, test_input = input_data[:train_size], input_data[train_size:]
train_target, test_target = target_data[:train_size], target_data[train_size:]

# Initialize LSM
lsm = LiquidStateMachine(reservoir_size=reservoir_size)

# Process the temperature data through the LSM
train_states = get_states(train_input)
test_states = get_states(test_input)

# Train a readout layer (Ridge Regression) for prediction
readout = Ridge(alpha=1.0)
readout.fit(train_states, train_target)

# Predict on test data
predictions = readout.predict(test_states)

# Evaluate the model
mse = mean_squared_error(test_target, predictions)
print(f"Mean Squared Error on Test Data: {mse:.4f}")

# Example of prediction
print(f"Example Prediction:\nTrue: {test_target[:5]}\nPredicted: {predictions[:5]}")


# In[9]:


plot_results(test_target, predictions)


# In[10]:


def new_data_and_retrain(new_num, i):
    # Additional temperature samples
    new_start = 2000+i*new_num
    new_data = generate_data(new_num, new_start)

    # Create input and target pairs for the new data
    new_input_data, new_target_data = create_pairs(new_data)

    # Process new data through the reservoir
    new_states = get_states(new_input_data)

    # Combine old and new data
    combined_train_states = np.vstack((train_states, new_states))
    combined_train_targets = np.hstack((train_target, new_target_data))

    # Retrain the readout layer with combined data
    readout.fit(combined_train_states, combined_train_targets)


# In[11]:


for i in range(200):
    new_data_and_retrain(new_num, i)

    # Predict on the test data again
    updated_predictions = readout.predict(test_states)

    # Evaluate the updated model
    mse_progress.append(mean_squared_error(test_target, updated_predictions))
    #print(f"Updated Mean Squared Error on Test Data: {mse_progress[i]:.4f}")


# In[12]:


mse_progress


# In[13]:


plot_results(test_target, updated_predictions)


# In[ ]:





# In[14]:


'''
# Train a readout layer (Ridge Regression) for prediction
readout.fit(train_states, train_target)

# Predict on test data
predictions = readout.predict(test_states)

# Evaluate the model
mse.append(mean_squared_error(test_target, predictions))
print(f"Mean Squared Error on Test Data: {mse[0]:.4f}")

# Example of prediction
print(f"Example Prediction:\nTrue: {test_target[:5]}\nPredicted: {predictions[:5]}")
'''

