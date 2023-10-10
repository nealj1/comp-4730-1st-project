# add graph for 
# heatmap
# edible vs poisonous
# we should do the cv like in the regression where we graph each of the fold values
# print the decision tree graph to see what the nodes it split on are
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import plot_tree
import pandas as pd

def print_class_compare(mushroom_data, title):
    # Count the frequency of each class
    class_counts = mushroom_data['class'].value_counts()

    # Plot the bar graph
    plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
    ax = class_counts.plot(kind='bar', color=['green', 'red'])  # Customize colors if desired

    # Define custom labels for the x-axis ticks
    custom_labels = ['EDIBLE', 'POISONOUS']  # Replace with your desired labels

    # Set the custom labels for the x-axis ticks
    plt.xticks(range(len(custom_labels)), custom_labels)

    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def print_compare_model_accuracies(model_names, accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Accuracy')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.show()

# Print Heatmap
def print_heatmap(mushroom_data):
    # check to get a rough idea of correlations
    correlations = mushroom_data.corr()
    plt.subplots(figsize=(20, 15))
    data_corr = sns.heatmap(correlations, annot=True, linewidths=0.5, cmap="RdBu_r")
    plt.show()

def print_crossfold_data(cv_model_names, fold_accuracies, mean_accuracies, num_folds):
    # Iterate through models to create individual plots
    for i, model_name in enumerate(cv_model_names):
        plt.figure(figsize=(10, 6))
        plt.title(f'Mean and Individual Fold Accuracies for {model_name}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')

        # Plot individual fold accuracies
        fold_accuracy = fold_accuracies[i]
        x_values = np.arange(1, num_folds + 1)
        plt.scatter(x_values, fold_accuracy, label=f'Fold Accuracies ({model_name})', marker='o')

        # Plot the mean accuracy
        mean_accuracy = mean_accuracies[i]
        plt.plot(x_values, [mean_accuracy] * num_folds, label=f'Mean Accuracy ({model_name})', linestyle='--')

        plt.xticks(x_values)
        plt.legend()
        plt.tight_layout()
        plt.show()


def print_crossfold_compare_all_same_graph(cv_model_names, fold_accuracies, mean_accuracies, num_folds):
    plt.figure(figsize=(10, 6))
    
    for i, model_name in enumerate(cv_model_names):
        # Plot individual fold accuracies
        fold_accuracy = fold_accuracies[i]
        x_values = np.arange(1, num_folds + 1)
        plt.scatter(x_values, fold_accuracy, label=f'Fold Accuracies ({model_name})', marker='o')

        # Plot the mean accuracy
        mean_accuracy = mean_accuracies[i]
        plt.plot(x_values, [mean_accuracy] * num_folds, label=f'Mean Accuracy ({model_name})', linestyle='--')

    plt.title('Mean and Individual Fold Accuracies for Models')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(x_values)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_crossfold_compare_separate(cv_model_names, fold_accuracies, mean_accuracies, num_folds):
    figsize=(10, 6)
    num_models = len(cv_model_names)
    
    # Define default colors
    colors = ['b', 'g', 'r', 'c', 'm']  # You can customize these colors
    
    fig, axes = plt.subplots(num_models, 1, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True)
    
    for i, model_name in enumerate(cv_model_names):
        # Plot individual fold accuracies
        fold_accuracy = fold_accuracies[i]
        x_values = np.arange(1, num_folds + 1)
        axes[i].scatter(x_values, fold_accuracy, label=f'Fold Accuracies ({model_name})', marker='o', color=colors[i])

        # Plot the mean accuracy
        mean_accuracy = mean_accuracies[i]
        axes[i].plot(x_values, [mean_accuracy] * num_folds, label=f'Mean Accuracy ({model_name})', linestyle='--', color=colors[i])

        axes[i].set_title(f'Mean and Individual Fold Accuracies for {model_name}')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()

    plt.xlabel('Fold')
    plt.tight_layout()
    plt.show()


def print_decision_tree_classifer(X, cv_model_names, models):
    # Define the name of the model you want to visualize
    model_name_to_visualize = "DecisionTreeClassifier"

    # Find the index of the model with the specified name in the list of model names
    model_index = cv_model_names.index(model_name_to_visualize)

    # Plot the decision tree for the specified model
    plt.figure(figsize=(10, 6))
    plot_tree(models[model_index], filled=True, feature_names=list(X.columns), class_names=["EDIBLE", "POISONOUS"])
    plt.title(f"Decision Tree for {model_name_to_visualize}")
    plt.show()

# LOGISTIC REGRESSION MODEL VISUAL
def print_logistic_regression_visual(X, models):
    # After the loop, you can access the logistic_regression model
    logistic_regression = models[1]  # Assuming logistic regression is the second model in the list

    # Now you can access the coefficients of the logistic_regression model
    coefficients = logistic_regression.coef_[0]

    # Match coefficients with feature names
    feature_names = list(X.columns)

    # Create a DataFrame to store feature names and their corresponding coefficients
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Sort the DataFrame by coefficient values (absolute values for magnitude)
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])

    # Sort the DataFrame by absolute coefficient values (in descending order)
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

    # Create a figure and axis for the bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Coefficient', y='Feature', data=coef_df)

    # Define a color palette with a unique color for each bar
    num_bars = len(coef_df)
    color_palette = sns.color_palette("husl", num_bars)  # You can choose any color palette you like

    # Assign a different color to each bar
    for i, bar in enumerate(ax.patches):
        bar.set_color(color_palette[i])

    # Customize the chart labels and title
    plt.title('Top Features - Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')

    # Show the plot
    plt.show()

def print_random_forest_classifier(X, models):
    randomforest = models[0]

    # Get feature importances from the model
    importances = randomforest.feature_importances_

    # Match importances with feature names
    feature_names = list(X.columns)

    # Create a DataFrame to store feature names and their corresponding importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort the features by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Create a custom color palette for the bars
    num_features = len(importance_df)

    # Create a figure and axis for the bar chart, specifying the custom color palette
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df)

    # Customize the chart labels and title
    plt.title('Feature Importances - RandomForestClassifier')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    # Show the plot
    plt.show()

def print_results_for_each_model(cv_model_names, fold_accuracies, fold_confusions, fold_reports, mean_accuracies):
    # Print the results for each fold and model at the end
    for model_name, fold_accuracy, fold_confusion, fold_report in zip(cv_model_names, fold_accuracies, fold_confusions, fold_reports):
        print(f"Model: {model_name}")
        for fold_num, (accuracy, confusion, report) in enumerate(zip(fold_accuracy, fold_confusion, fold_report), start=1):
            print(f"Fold {fold_num} Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:\n", confusion)
            print("Classification Report:\n", report)
            print("\n")
        print(f"Mean Accuracy: {mean_accuracies[cv_model_names.index(model_name)]:.4f}\n")

def print_grid_search_result(model_name, best_parameters, best_accuracy):
    # Print the results
    print(f"Best parameters for {model_name}: {best_parameters}")
    print(f"Best accuracy for {model_name}: {best_accuracy:.4f}")
    print("\n")

def print_grid_search(best_model_names, best_parameters_list):
    for model_name, best_parameters in zip(best_model_names, best_parameters_list):
        print(f"Model: {model_name} Best parameters: {best_parameters}")

def print_best_grid_training_results(model_name, accuracy, confusion, report):
    # Print the results for each model
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)
    print("\n")