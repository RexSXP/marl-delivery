import matplotlib.pyplot as plt
import numpy as np

def visualize_map(map_file):
    # Read the map file
    with open(map_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the map data (skipping the filepath comment line)
    map_data = []
    for line in lines:
        if '//' not in line:  # Skip comment lines
            row = [int(cell) for cell in line.strip().split()]
            if row:  # Only append non-empty rows
                map_data.append(row)
    
    map_array = np.array(map_data)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(map_array, cmap='binary')
    
    # Add grid lines
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, map_array.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, map_array.shape[0], 1), [])
    
    # Add title and labels
    plt.title('Map Visualization')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    # Call the function with your map file
    visualize_map('maps/map5.txt')