#########################################################################
####Read the saved estimated parameters and plot them out across time####
#########################################################################
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Define the list of file paths
# file_paths = ['/Users/mac/Downloads/tauo_estimates/run 0_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 1_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 2_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 3_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 4_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 5_tauo.dat',
#               '/Users/mac/Downloads/tauo_estimates/run 6_tauo.dat']

# Define the directory path and the common pattern in the filenames
directory = '/Users/mac/Downloads/taud_forwardfirst/'
filename_pattern = 'run %d_taud.dat'

# Specify the parameter name and give the ground truth value
parameter_name = "taud"
GT = 0.125

# Define the range of indices for the filenames
start_index = 0
end_index = 2

# Initialize an empty list to store the file paths
file_paths = []

# Generate the file paths based on the pattern and indices
for i in range(start_index, end_index+1):
    file_path = directory + filename_pattern % i
    file_paths.append(file_path)

# Set up the plot
plt.figure()

# Iterate over the files and plot the values
for file_path in file_paths:
    # Read the .dat file
    data = np.loadtxt(file_path, dtype=str)

    # Extract the epochs from the first column
    epochs = data[:, 0].astype(int)

    # Extract the values from the second column and remove square brackets
    values_str = np.char.strip(data[:, 1], "[]")

    # Convert the values to float
    values = values_str.astype(float)
    
    # Get the initial value
    # ini_val = values[0]
    # print(ini_val)

    # # Get the final estimate value
    final_estimate = values[-1]
    print(final_estimate)

    # Get the file name without extension as the label
    label = file_path.split('/')[-1].split('.')[0]

    # Plot the values as a function of epochs with the label
    plt.plot(epochs, values, '--', label=label)

# Plot the GT value
plt.plot(epochs, np.full(len(epochs), GT), 'k--')
# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Estimated parameter')
plt.title(f'Estimated {parameter_name} vs. Epochs')
plt.legend()
plt.show()

# Save the final estimates

##########################################################################################
##########################################################################################

#%% if only one file to plot
# Define the directory path and the common pattern in the filenames
directory = '/Users/mac/Downloads/'
filename = 'run 0_tausi.dat'

# Specify the parameter name
parameter_name = "taud"

file_path = directory + filename

# Set up the plot
plt.figure()

# Read the .dat file
data = np.loadtxt(file_path, dtype=str)

# Extract the epochs from the first column
epochs = data[:, 0].astype(int)
# Extract the values from the second column and remove square brackets
values_str = np.char.strip(data[:, 1], "[]")
# Convert the values to float
values = values_str.astype(float)

# Get the initial value
ini_val = values[0]
print(ini_val)
# # Get the final estimate value
final_estimate = values[-1]
print(f"final estimates: {final_estimate}")
# Get the file name without extension as the label
label = file_path.split('/')[-1].split('.')[0]
# Plot the values as a function of epochs with the label
plt.plot(epochs, values, '--', label=label)

# Plot the GT value
plt.plot(epochs, np.full(len(epochs),114), 'k--')
# Add labels, title, and legend
plt.xlabel('Epochs')
plt.ylabel('Estimated parameter')
plt.title(f'Estimated {parameter_name} vs. Epochs')
# plt.legend()
plt.show()

# %%
