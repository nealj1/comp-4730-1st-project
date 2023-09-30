
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_model_performance(model_names, train_metrics, test_metrics, metric_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = range(len(model_names))

    train_bar = ax.bar([pos - width/2 for pos in x], train_metrics, width, label='Training')
    test_bar = ax.bar([pos + width/2 for pos in x], test_metrics, width, label='Testing')

    ax.set_xlabel('Models')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def print_metrics(model_metrics):
    for metrics in model_metrics:
        print(metrics['Model Name'])
        print("Training MSE:", metrics['Training MSE'])
        print("Testing MSE:", metrics['Testing MSE'])
        print("Training RMSE:", metrics['Training RMSE'])
        print("Testing RMSE:", metrics['Testing RMSE'])
        print("Training MAE:", metrics['Training MAE'])
        print("Testing MAE:", metrics['Testing MAE'])
        print("Training R2:", metrics['Training R2'])   
        print("Testing R2:", metrics['Testing R2'])
        print()

def print_comparision(df_metrics):
    # Create a bar chart comparing RMSE values
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model Name', y='Testing RMSE', data=df_metrics)
    plt.xlabel('Model Name')
    plt.ylabel('Testing Root Mean Squared Error (RMSE)')
    plt.title('Model Comparison - Testing RMSE')
    plt.xticks(rotation=45)
    plt.show()