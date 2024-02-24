import matplotlib
# Set the backend to 'Agg' to avoid GUI-related errors
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Create a simple plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot")

# Save the plot to a file
plt.savefig("test_plot.png")

print("Plot saved as 'test_plot.png'. Check your current directory.")