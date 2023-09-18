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
filenames = ["Adelaide_Data","Perth_Data", "Sydney_Data", "Tasmania_Data"]

#PRINTING
#----------------------------------------------------------------------------------------------------------------------------------------------
def print_total_heatmap_plot():
    # Create a scatter plot for all data points
    plt.scatter(X.values.flatten(), y.values.flatten(), c=po.values.flatten(), cmap='viridis', label='Data Points')

    # Add labels and a colorbar
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')
    plt.colorbar(label='Z-Axis (Color)')

    # Add a title and legend (if needed)
    plt.title('Heatmap')
    plt.legend()

    # Show the plot
    plt.show()
    return

def print_single_heatmap_plot(postion):
    # Create a scatter plot for all data points
    plt.scatter(X.loc[postion], y.loc[postion], c=po.loc[postion], cmap='viridis', label='Power Output')

    # Add labels and a colorbar
    plt.xlabel('X-Axis (Long.)')
    plt.ylabel('Y-Axis (Lat.)')
    plt.colorbar(label='Z-Axis (Color)')

    # Add a title and legend (if needed)
    plt.title('Heatmap of Single Configuration')
    plt.legend()

    # Show the plot
    plt.show()
    return

# Define a function to create the subplot with scatter plots
def create_subplot_scatter_plots(dataframe, rows, columns):
    fig, axs = plt.subplots(rows, columns, figsize=(12, 12))

    for pos, ax in enumerate(axs.ravel()):
        if pos >= len(dataframe):
            break

        x_data = dataframe.loc[pos, 'X1':'X9']
        y_data = dataframe.loc[pos, 'Y1':'Y9']
        po_data = dataframe.loc[pos, 'P1':'P9']

        sc = ax.scatter(x_data, y_data, c=po_data, cmap='viridis', label='Power Output')
        ax.set_xlabel('X-Axis (Long.)')
        ax.set_ylabel('Y-Axis (Lat.)')
        ax.set_title(f'Heatmap Plot for Row {pos} of {filenames[0]}')
        ax.legend()

        # Add colorbar to each subplot
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Z-Axis (Color)')

    plt.tight_layout()
    plt.show()
    return

# Create a scatter plot of the individual power on a certain configuration, a 16x16 grid with a certain wave pattern setup, and the power generated in there
def print_scatter_plot(pos):
    plt.scatter(X.loc[pos], y.loc[pos], cmap='viridis', label='WEC Locations')

    # Add labels and a colorbar
    plt.xlabel('X-Axis (Long.)')
    plt.ylabel('Y-Axis (Lat.)')
    # Add a title and legend (if needed)
    plt.title('Distinct Configuration of the Wave Farm')
    plt.legend()

    # Show the plot
    plt.show()
    return

# Create a bar plot
def print_bar_test():
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
    return



#---------------------------------------------------------------------------------------------------------------------------------------------- End Printing

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
po = df[[f'P{i}' for i in range(1, power_output)]]
# Select 'Powerall' column
pa = df[['Powerall']]

# Print Graphs
#print_total_heatmap_plot()
#print_scatter_plot(1)
print_single_heatmap_plot(1)
# Call the function to create the subplot with scatter plots for the first 9 rows
#create_subplot_scatter_plots(df[:9], 3, 3)

print(f'{type(X)} : {len(X)} : {X.shape}')
print(f'{type(y)} : {len(y)} : {y.shape}')
print(f'{type(po)} : {len(po)} : {po.shape}')
print(f'{type(pa)} : {len(pa)} : {pa.shape}')


# Process the data
    # Split in to training set and test set
    # Determine representation of input
    #Determine the representatin of the output
# Choose form of model: ex. linear regression
# Decide how to evalutate the system's performance: objective function
# Set model parameters to optimize performance
# Evalute on test set: generalization

# Choose a dataset with at least 5000 instances and 20 attributes for classification or regression. Compare how the different approaches seen in class perform on this dataset to predict accurately the classes or the values of the unlabeled data. You should determine what are the best hyper-parameters for each approach you are using. 


