
import matplotlib.pyplot as plt

def print_total_heatmap_plot(X,y,po):
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
def create_subplot_scatter_plots(dataframe, rows, columns, filename):
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
        ax.set_title(f'Heatmap Plot for Row {pos} of {filename}')
        ax.legend()

        # Add colorbar to each subplot
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Z-Axis (Color)')

    plt.tight_layout()
    plt.show()
    return

# Create a scatter plot of the individual power on a certain configuration, a 16x16 grid with a certain wave pattern setup, and the power generated in there
def print_scatter_plot(X,y,pos):
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
