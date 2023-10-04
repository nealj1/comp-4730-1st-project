import matplotlib.pyplot as plt

# Sample data
x_axis = [1, 2, 3, 4, 5]
y_axis = [10, 20, 30, 40, 50]
z_axis = [0.5, 0.8, 0.2, 0.7, 0.9]

# Combine the lists element-wise using zip
combined = zip(x_axis, y_axis, z_axis)

# Unzip the combined data
x_combined, y_combined, z_combined = zip(*combined)

# Create a scatter plot
plt.scatter(x_combined, y_combined, c=z_combined, cmap='viridis', label='Data Points')

# Add labels and a colorbar
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.colorbar(label='Z-Axis (Color)')

# Add a title and legend (if needed)
plt.title('Scatter Plot with Color by Z-Axis')
plt.legend()

# Show the plot
plt.show()
