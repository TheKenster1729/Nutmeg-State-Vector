import pickle
import tensorflow as tf
import numpy as np
from MDN_model import TrainModel
import pandas as pd
import matplotlib.pyplot as plt
import random
import contextily as ctx
from matplotlib.colors import to_rgba

def plot_random_prediction():
    model = TrainModel()
    model.load_model()
    test_df = pd.read_csv("test_set_streets.csv")

    # Select a random row from the test dataset
    random_idx = random.randint(0, len(test_df) - 1)
    random_street = test_df.iloc[random_idx]
    
    # Extract street name and actual coordinates
    street_name = random_street['street_name']
    actual_lon = random_street['lon']
    actual_lat = random_street['lat']
    
    # Make prediction using the model
    predicted_coords = model.predict_location(street_name)
    predicted_lon, predicted_lat = predicted_coords[0], predicted_coords[1]
    
    # Create a new figure for plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate distance error (in coordinate units)
    error = np.sqrt((actual_lon - predicted_lon)**2 + (actual_lat - predicted_lat)**2)
    
    # Plot actual and predicted locations
    ax.scatter(actual_lon, actual_lat, color='blue', label='Actual Location', s=150, marker='o', 
               edgecolor='black', zorder=3)
    ax.scatter(predicted_lon, predicted_lat, color='red', label='Predicted Location', s=150, marker='x', 
               linewidth=3, zorder=3)
    
    # Draw a line connecting the points to show the error
    ax.plot([actual_lon, predicted_lon], [actual_lat, predicted_lat], 'k--', alpha=0.7, linewidth=2, zorder=2)
    
    # Set the extent of the map to show Connecticut
    # Adjust the buffer to make sure both points are visible
    buffer = max(error * 2, 0.05)  # At least some buffer to see surrounding area
    min_lon = min(actual_lon, predicted_lon) - buffer
    max_lon = max(actual_lon, predicted_lon) + buffer
    min_lat = min(actual_lat, predicted_lat) - buffer
    max_lat = max(actual_lat, predicted_lat) + buffer
    
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add the basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Add labels and title
    plt.title(f"Street: '{street_name}'\nPrediction Error: {error:.6f} coordinate units", fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add text annotations for the coordinates with background for better visibility
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    
    ax.annotate(f"Actual: ({actual_lon:.6f}, {actual_lat:.6f})", 
                xy=(actual_lon, actual_lat), 
                xytext=(15, 15), 
                textcoords='offset points',
                bbox=bbox_props,
                zorder=4)
    
    ax.annotate(f"Predicted: ({predicted_lon:.6f}, {predicted_lat:.6f})", 
                xy=(predicted_lon, predicted_lat), 
                xytext=(15, 15), 
                textcoords='offset points',
                bbox=bbox_props,
                zorder=4)
    
    # Add legend with background for better visibility
    legend = ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    
    plt.tight_layout()
    plt.show()
    
    return street_name, (actual_lon, actual_lat), (predicted_lon, predicted_lat), error

def plot_street_locations(street_name):
    """
    Plot all locations with a given street name and the model's prediction.
    
    Args:
        street_name (str): The street name to search for and predict
    """
    model = TrainModel()
    model.load_model()
    
    # Load the full dataset to find all occurrences of the street name
    # Using the full dataset instead of just the test dataset
    full_df = pd.read_csv("full_dataset.csv")
    
    # Filter for the given street name
    street_locations = full_df[full_df['street_name'] == street_name]
    
    if len(street_locations) == 0:
        print(f"No locations found with street name: '{street_name}'")
        return
    
    # Make prediction using the model
    predicted_coords = model.predict_location(street_name)
    predicted_lon, predicted_lat = predicted_coords[0], predicted_coords[1]
    
    # Create a new figure for plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all actual locations
    for idx, location in street_locations.iterrows():
        actual_lon = location['lon']
        actual_lat = location['lat']
        
        # Calculate distance from prediction to this actual location
        error = np.sqrt((actual_lon - predicted_lon)**2 + (actual_lat - predicted_lat)**2)
        
        # Plot actual location with transparency based on count (for overlapping points)
        ax.scatter(actual_lon, actual_lat, color='blue', s=150, marker='o', 
                   edgecolor='black', alpha=0.7, zorder=3)
        
        # Add text annotations for the coordinates
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        ax.annotate(f"({actual_lon:.6f}, {actual_lat:.6f})", 
                    xy=(actual_lon, actual_lat), 
                    xytext=(15, 15), 
                    textcoords='offset points',
                    bbox=bbox_props,
                    fontsize=8,
                    zorder=4)
    
    # Plot the predicted location
    ax.scatter(predicted_lon, predicted_lat, color='red', label='Predicted Location', 
               s=200, marker='x', linewidth=3, zorder=5)
    
    # Add annotation for predicted location
    ax.annotate(f"Predicted: ({predicted_lon:.6f}, {predicted_lat:.6f})", 
                xy=(predicted_lon, predicted_lat), 
                xytext=(15, 15), 
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=10,
                zorder=6)
    
    # Connect all actual locations to the prediction with dashed lines
    for idx, location in street_locations.iterrows():
        actual_lon = location['lon']
        actual_lat = location['lat']
        ax.plot([actual_lon, predicted_lon], [actual_lat, predicted_lat], 
                'k--', alpha=0.5, linewidth=1, zorder=2)
    
    # Set the extent of the map to show all points
    all_lons = street_locations['lon'].tolist() + [predicted_lon]
    all_lats = street_locations['lat'].tolist() + [predicted_lat]
    
    # Calculate buffer based on the spread of points
    lon_range = max(all_lons) - min(all_lons)
    lat_range = max(all_lats) - min(all_lats)
    buffer = max(lon_range, lat_range) * 0.2  # 20% buffer
    buffer = max(buffer, 0.05)  # Minimum buffer
    
    ax.set_xlim(min(all_lons) - buffer, max(all_lons) + buffer)
    ax.set_ylim(min(all_lats) - buffer, max(all_lats) + buffer)
    
    # Add the basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Add labels and title
    plt.title(f"All locations for street: '{street_name}' ({len(street_locations)} occurrences)", 
              fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markeredgecolor='black', markersize=12, label='Actual Locations'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
               markeredgecolor='red', markersize=12, label='Predicted Location')
    ]
    
    # Add legend with background for better visibility
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                       frameon=True, framealpha=0.9)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    
    plt.tight_layout()
    plt.show()
    
    # Return information about the prediction
    return street_name, street_locations[['lon', 'lat']].values, (predicted_lon, predicted_lat)

# Example usage
if __name__ == "__main__":
    # You can either use the random prediction function
    # street_name, actual_coords, predicted_coords, error = plot_random_prediction()
    
    # Or plot all locations for a specific street name
    street_name = "New Haven Avenue"  # Replace with any street name you want to analyze
    plot_street_locations(street_name)

