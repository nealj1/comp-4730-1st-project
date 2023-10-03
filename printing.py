
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def print_compare_5fold_bargraph(cv_rmse_scores_dict):
    # List of metrics
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']

    # Extract model names and metric means
    model_names = list(cv_rmse_scores_dict.keys())
    metric_means = {metric: [model_data[metric]['mean'] for model_data in cv_rmse_scores_dict.values()] for metric in metrics}

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the width of each bar
    bar_width = 0.15
    index = np.arange(len(model_names))

    # Create bars for each metric
    for i, metric in enumerate(metrics):
        ax.bar(index + i * bar_width, metric_means[metric], bar_width, label=metric)

    # Configure plot
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Metric Value')
    ax.set_title('Mean Cross-Validation Metrics by Model')
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

def print_all_graphs_individual_cs_scores(cv_metrics_dict):
    # Iterate over models and create separate graphs for each model
    for model_name, model_data in cv_metrics_dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric, metric_data in model_data.items():
            scores = metric_data['scores']
            mean_score = metric_data['mean']
            
            ax.plot(np.arange(1, len(scores) + 1), scores, label=f'{metric} (Individual)')
            ax.axhline(y=mean_score, color='red', linestyle='--', label=f'{metric} (Mean)')
        
        # Configure graph
        ax.set_title(f'{model_name} - Cross-Validation Scores')
        ax.set_xlabel('Cross-Validation Fold')
        ax.set_ylabel('Score')
        ax.legend()
        
        plt.tight_layout()
        plt.show()

def print_cross_validation_data(cv_metrics_dict):
    # Print cross-validation metrics for each model
    for model_name, metrics_dict in cv_metrics_dict.items():
        print(f"Model: {model_name}")
        for metric_name, metric_values in metrics_dict.items():
            print(f"Metric: {metric_name}")
            print(f"Mean {metric_name}: {metric_values['mean']}")
            print(f"Individual {metric_name} scores: {metric_values['scores']}\n")
