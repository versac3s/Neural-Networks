# Step 1: Generate a vector of inputs and a vector of weights
import numpy as np

# Set the seed for reproducibility
np.random.seed(seed=0)

# Generate a random input vector I, sampling from {0,1}
I = np.random.choice([0,1], 3)

# Generate a random weight vector W, sampling from {-1,1}
W = np.random.choice([-1,1], 3)

# Display the input and weight vectors
print(f'Input vector: {I}, Weight vector: {W}')

# Step 2: Compute the dot product between the input and weight vectors
dot = I @ W

# Display the dot product result
print(f'Dot product: {dot}')

# Step 3: Define the threshold activation function
def linear_threshold_gate(dot: int, T: float) -> int:
    '''
    Computes the output of a linear threshold gate.
    
    Parameters:
    dot (int): The dot product of inputs and weights.
    T (float): The threshold value.
    
    Returns:
    int: Binary output based on the threshold.
    '''
    if dot >= T:
        return 1
    else:
        return 0

# Step 4: Compute the output based on the specified threshold value
T = 1  # Threshold value
activation = linear_threshold_gate(dot, T)

# Display the activation result
print(f'Activation: {activation}')

# Application: Boolean algebra using the McCulloch-Pitts artificial neuron

# The AND Function

# Step 1: Generate a matrix of inputs and a vector of weights for the AND function
input_table = np.array([
    [0, 0],  # both inputs are no (0)
    [0, 1],  # one input no, one input yes
    [1, 0],  # one input yes, one input no
    [1, 1]   # both inputs are yes (1)
])

# Display the input table
print(f'Input table:\n{input_table}')

# Define the weight vector for the AND function
weights = np.array([1, 1])

# Display the weight vector
print(f'Weights: {weights}')

# Step 2: Compute the dot product between the input matrix and the weight vector
dot_products = input_table @ weights

# Display the dot product results
print(f'Dot products: {dot_products}')

# Step 3: Compute the output based on the threshold value
T = 2  # Threshold value for AND function

# Iterate over the dot product results to compute activations
for i in range(0, 4):
    activation = linear_threshold_gate(dot_products[i], T)
    
    # Display the activation result for each input
    print(f'Activation for input {input_table[i]}: {activation}')
