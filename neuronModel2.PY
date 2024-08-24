# Practical-2: Realization of OR, NOT and NOR functions using McCulloch-Pitts Neuron Model and display of decision boundary.
import numpy as np
import matplotlib.pyplot as plt

# The OR Function

# Step 1: Generate a vector of inputs and a vector of weights
# Matrix of inputs
input_table = np.array([
    [0, 0],  # both no
    [0, 1],  # one no, one yes
    [1, 0],  # one yes, one no
    [1, 1]   # both yes
])
print(f'input table:\n{input_table}')

# Array of weights
weights = np.array([1, 1])
print(f'weights: {weights}')

# Step 2: Compute the dot product between the matrix of inputs and weights
# Dot product matrix of inputs and weights
dot_products = input_table @ weights
print(f'Dot products: {dot_products}')

# Step 3: Define the threshold activation function
def linear_threshold_gate(dot: int, T: float) -> int:
    '''Returns the binary threshold output'''
    if dot >= T:
        return 1
    else:
        return 0

# Step 4: Compute the output based on the threshold value
T = 1
for i in range(0, 4):
    activation = linear_threshold_gate(dot_products[i], T)
    print(f'Activation: {activation}')

# Visualization of OR function decision boundary
plt.figure(figsize=(8, 6))
for i in range(len(input_table)):
    if dot_products[i] >= T:
        plt.scatter(input_table[i][0], input_table[i][1], color='blue', label='OR=1' if i == 0 else "")
    else:
        plt.scatter(input_table[i][0], input_table[i][1], color='red', label='OR=0' if i == 0 else "")

# Plotting the decision boundary for OR function
x_vals = np.array([0, 1])
y_vals = (T - weights[0] * x_vals) / weights[1]
plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary (OR)')

plt.title("OR Function - Decision Boundary")
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.grid(True)
plt.show()

# The NOR Function

# Step 1: Generate a vector of inputs and a vector of weights
# Array of weights
weights = np.array([-1, -1])
print(f'weights: {weights}')

# Step 2: Compute the dot product between the matrix of inputs and weights
# Dot product matrix of inputs and weights
dot_products = input_table @ weights
print(f'Dot products: {dot_products}')

# Step 3: Compute the output based on the threshold value
T = 0
for i in range(0, 4):
    activation = linear_threshold_gate(dot_products[i], T)
    print(f'Activation: {activation}')

# Visualization of NOR function decision boundary
plt.figure(figsize=(8, 6))
for i in range(len(input_table)):
    if dot_products[i] >= T:
        plt.scatter(input_table[i][0], input_table[i][1], color='blue', label='NOR=1' if i == 0 else "")
    else:
        plt.scatter(input_table[i][0], input_table[i][1], color='red', label='NOR=0' if i == 0 else "")

# Plotting the decision boundary for NOR function
x_vals = np.array([0, 1])
y_vals = (T - weights[0] * x_vals) / weights[1]
plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary (NOR)')

plt.title("NOR Function - Decision Boundary")
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.legend()
plt.grid(True)
plt.show()

# The NOT Function

# Step 1: Generate a vector of inputs and a vector of weights
input_table_not = np.array([
    [0],  # no
    [1]   # yes
])
print(f'input table:\n{input_table_not}')

# Array of weights for NOT function
weights_not = np.array([-1])
print(f'weights: {weights_not}')

# Step 2: Compute the dot product between the matrix of inputs and weights
dot_products_not = input_table_not @ weights_not
print(f'Dot products: {dot_products_not}')

# Step 3: Compute the output based on the threshold value
T_not = 0
activations_not = []
for i in range(0, 2):
    activation_not = linear_threshold_gate(dot_products_not[i], T_not)
    activations_not.append(activation_not)
    print(f'Activation: {activation_not}')

# Visualization of NOT function decision boundary
plt.figure(figsize=(8, 6))
for i in range(len(input_table_not)):
    if dot_products_not[i] >= T_not:
        plt.scatter(input_table_not[i][0], 0, color='blue', label='NOT=1' if i == 0 else "")
    else:
        plt.scatter(input_table_not[i][0], 0, color='red', label='NOT=0' if i == 0 else "")

# Plotting the decision boundary for NOT function
x_vals_not = np.array([0, 1])
y_vals_not = (T_not - weights_not[0] * x_vals_not) / 1  # This is essentially just a line parallel to x-axis
plt.plot(x_vals_not, y_vals_not, '--', color='green', label='Decision Boundary (NOT)')

plt.title("NOT Function - Decision Boundary")
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
