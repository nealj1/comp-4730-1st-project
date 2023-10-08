# add graph for 
# heatmap
# edible vs poisonous
# we should do the cv like in the regression where we graph each of the fold values
# print the decision tree graph to see what the nodes it split on are
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Show the bar chart
    plt.show()

def print_heatmap(mushroom_data):
    # check to get a rough idea of correlations
    correlations = mushroom_data.corr()
    plt.subplots(figsize=(20, 15))
    data_corr = sns.heatmap(correlations, annot=True, linewidths=0.5, cmap="RdBu_r")
    plt.show()