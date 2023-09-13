# MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# GLOBAL VARIABLES
X_grid_size = 17
y_grid_size = 17
power_output = 17

#PRINTING
#------------------------------------------------------------------------
# PRINT TEST
def print_test():
    # Create a scatter plot
    plt.scatter(X, y, label='Data Points')

    # Add labels and a title
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.title('Scatter Plot Example')

    # Add a legend (if needed)
    plt.legend()

    # Show the plot
    plt.show()
    return


# Create a 3D scatter plot
def scatter3dplot_test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_values, y_values, p_values, c='r', marker='o', label='P')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')
    ax.set_title('3D Scatter Plot')
    plt.legend()

    # Show the plot
    plt.show()

# Create a figure and 3D subplot
def figand3dsubplot_test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D mesh plot
    ax.plot_trisurf(x_values, y_values, p_values, cmap='viridis')

    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')
    ax.set_title('3D Mesh Plot')

    # Show the plot
    plt.show()

def print_test():
    # Sample data (replace with your actual data)
    o_values = o_value * len(x_values)  # Replace with your actual values

    # Create a figure and 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Create a colormap for the heights (you can choose any colormap you like)
    #cmap = cm.get_cmap('viridis')
    cmap = mpl.colormaps['viridis']

    # Normalize the Z-values to fit the colormap range
    norm = plt.Normalize(min(o_values), max(o_values))

    # Create a 3D mesh plot with color mapping
    sc = ax.plot_trisurf(x_values, y_values, o_values, cmap=cmap, norm=norm)

    # Add colorbar to show height values
    fig.colorbar(sc, ax=ax, label='Height')

    # Label the axes
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('O')
    ax.set_title('3D Mesh Plot')

    # Show the plot
    plt.show()


# Create a bar plot
def print_bar():
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Bar plot for X values
    axs[0].bar(range(len(x_values)), x_values)
    axs[0].set_xlabel('X Values')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Bar Plot of X Values')

    # Bar plot for Y values
    axs[1].bar(range(len(y_values)), y_values)
    axs[1].set_xlabel('Y Values')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Bar Plot of Y Values')

    # Bar plot for P values
    axs[2].bar(range(len(p_values)), p_values)
    axs[2].set_xlabel('P Values')
    axs[2].set_ylabel('Value')
    axs[2].set_title('Bar Plot of P Values')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

# Plot lines
def crazy_lines():
    plt.plot(x_values, y_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Plot of X vs. Y')
    plt.show()



#------------------------------------------------------------------------

# DATA
# Create a DataFrame for the data
df = pd.read_csv('WECs_DataSet/Adelaide_Data.csv', header=None)

# X1, X2, ..., X16, Y1, Y2, ..., Y16, P1, P2, ..., P16, Powerall
df.columns = [f'X{i}' for i in range(1, X_grid_size)] +[f'Y{i}' for i in range(1, y_grid_size)]+ [f'P{i}' for i in range(1, power_output)] + ['Powerall']

# Select 'X' columns
X = df[[f'X{i}' for i in range(1, X_grid_size)]]
# Select 'Y' columns
y = df[[f'Y{i}' for i in range(1, y_grid_size)]]
# Select 'P' columns
p = df[[f'P{i}' for i in range(1, power_output)]]
# Select 'Powerall' column
o = df[['Powerall']]

# Process the data
    # Split in to training set and test set
    # Determine representation of input
    #Determine the representatin of the output
# Choose form of model: ex. linear regression
# Decide how to evalutate the system's performance: objective function
# Set model parameters to optimize performance
# Evalute on test set: generalization

# Choose a dataset with at least 5000 instances and 20 attributes for classification or regression. Compare how the different approaches seen in class perform on this dataset to predict accurately the classes or the values of the unlabeled data. You should determine what are the best hyper-parameters for each approach you are using. 


# Iterate over rows and print X and Y values
for index in range(len(df)):
    row = df.iloc[index]
    x_values = [row[f'X{i}'] for i in range(1, X_grid_size)]
    y_values = [row[f'Y{i}'] for i in range(1, y_grid_size)]
    p_values = [row[f'P{i}'] for i in range(1, power_output)]
    o_value   = [row[f'Powerall']]
    #print(f"X: {x_values}, Y: {y_values}, P: {p_values}, O: {o_value}")
