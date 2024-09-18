import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import math

# Load the CSV files
df_polygons = pd.read_csv('Dataset/Random_SiteVertices.csv')  # ‘TARGET_FID’, ‘POINT_X’, ‘POINT_Y’, ‘HEIGHT’
df_building_info = pd.read_csv('Dataset/Random_BuildingInformation.csv', index_col='ORIG_FID')  # 'ORIG_FID', 'SIA2024BuildingType', 'BuildingAge', 'GlazingRatio'

# Group the data by TARGET_FID
grouped = df_polygons.groupby('TARGET_FID')

# Initialize lists to store results
results = []

# Function to compute the orientation angle relative to the y-axis
def compute_orientation_angle(polygon, shorter_side_length):
    x, y = polygon.exterior.xy
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        if np.isclose(segment_length, shorter_side_length, rtol=1e-2):
            angle_with_x = math.degrees(math.atan2(dy, dx))
            angle_with_y = 90 - angle_with_x
            return angle_with_y
    raise ValueError("Shorter side length not found in the polygon.")

# Function to compute exposition based on orientation angle
def compute_exposition(angle):
    if angle < 0:
        angle = 360 + angle
    if (337.5 <= angle < 360) or (0 <= angle < 22.5) or (157.5 <= angle < 202.5):
        return 'N-S'
    elif (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
        return 'NE-SW'
    elif (247.5 <= angle < 292.5) or (67.5 <= angle < 112.5):
        return 'W-E'
    elif (292.5 <= angle < 337.5) or (112.5 <= angle < 157.5):
        return 'NW-SE'
    else:
        raise ValueError("Invalid angle for exposition calculation.")

# Function to bin continuous values
def bin_values(values, num_bins=10):
    values = np.array(values)
    bin_edges = np.linspace(values.min(), values.max(), num_bins)
    bin_indices = np.digitize(values, bins=bin_edges, right=True)
    return bin_indices

# Process each group (cuboid)
for target_fid, group in grouped:
    coords = group[['POINT_X', 'POINT_Y']].values
    if len(coords) != 5 or not np.array_equal(coords[0], coords[-1]):
        raise ValueError(f"Invalid coordinates for TARGET_FID {target_fid}")
    polygon = Polygon(coords[:-1])

    area = polygon.area

    x, y = np.array(polygon.exterior.xy)
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    distances = np.unique(distances)

    width = min(distances)
    length = max(distances)

    #aspect_ratio = length / width

    orientation_angle = compute_orientation_angle(polygon, width if width < length else length)
    orientation_angle = 360 + orientation_angle if orientation_angle < 0 else orientation_angle
    exposition = compute_exposition(orientation_angle)

    glazing_ratio = df_building_info.loc[target_fid, 'GlazingRatio']
    building_age = df_building_info.loc[target_fid, 'BuildingAge']

    results.append([target_fid, area, width, length, exposition, glazing_ratio, building_age])

# Create a DataFrame from the resultsimport pandas as pd
import numpy as np
from shapely.geometry import Polygon
import math

# Load the CSV file
file_path = 'Dataset/Random_SiteVertices.csv'

df_polygons = pd.read_csv('Dataset/Random_SiteVertices.csv') #‘TARGET_FID’, ‘POINT_X’, POINT_Y’, ‘HEIGHT’

# Group the data by TARGET_FID
grouped = df_polygons.groupby('TARGET_FID')

df_building_info = pd.read_csv('Dataset/Random_BuildingInformation.csv',index_col='ORIG_FID') #'ORIG_FID', 'SIA2024BuildingType','BuildingAge', 'GlazingRatio'


# Initialize lists to store results
areas = []
widths = []
lengths = []
aspect_ratios = []
orientations = []
expositions = []

# Function to compute the orientation angle relative to the y-axis
def compute_orientation_angle(polygon, shorter_side_length):
    # Get the coordinates of the base rectangle
    x, y = polygon.exterior.xy

    # Find the segment of the polygon that matches the shorter side length
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        segment_length = np.sqrt(dx**2 + dy**2)

        if np.isclose(segment_length, shorter_side_length, rtol=1e-2):
            # Calculate the angle of this segment with respect to the x-axis
            angle_with_x = math.degrees(math.atan2(dy, dx))
            # Calculate the angle with respect to the y-axis
            angle_with_y = 90 - angle_with_x
            return angle_with_y

    raise ValueError("Shorter side length not found in the polygon.")

# Function to compute exposition based on orientation angle
def compute_exposition(angle):
    if angle < 0:
        angle = 360 + angle
    if (337.5 <= angle < 360) or (0 <= angle < 22.5) or (157.5 <= angle < 202.5):
        return 'N-S'
    elif (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
        return 'NE-SW'
    elif (247.5 <= angle < 292.5) or (67.5 <= angle < 112.5):
        return 'W-E'
    elif (292.5 <= angle < 337.5) or (112.5 <= angle < 157.5):
        return 'NW-SE'
    else:
        raise ValueError("Invalid angle for exposition calculation.")

# Process each group (cuboid)
for target_fid, group in grouped:
    # Extract coordinates and create a Polygon
    coords = group[['POINT_X', 'POINT_Y']].values
    if len(coords) != 5 or not np.array_equal(coords[0], coords[-1]):
        raise ValueError(f"Invalid coordinates for TARGET_FID {target_fid}")
    polygon = Polygon(coords[:-1])  # Exclude the repeated last point

    # Compute area
    area = polygon.area
    areas.append(area)

    # Compute width and length
    x, y = np.array(polygon.exterior.xy)
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    distances = np.unique(distances)

    width = min(distances)
    length = max(distances)

    widths.append(width)
    lengths.append(length)

    # Compute aspect ratio
    aspect_ratio = length / width
    aspect_ratios.append(aspect_ratio)

    # Compute orientation angle
    orientation_angle = compute_orientation_angle(polygon, width if width < length else length)
    print(orientation_angle)
    orientation_angle = 360 + orientation_angle if orientation_angle < 0 else orientation_angle
    print(orientation_angle)
    exposition = compute_exposition(orientation_angle)

    orientations.append(orientation_angle)
    expositions.append(exposition)

# Print the results
for i, target_fid in enumerate(grouped.groups.keys()):
    print(f"TARGET_FID {target_fid}:")
    print(f"  Area: {areas[i]}")
    print(f"  Width: {widths[i]}")
    print(f"  Length: {lengths[i]}")
    print(f"  Aspect Ratio: {aspect_ratios[i]}")
    print(f"  Orientation Angle: {orientations[i]}")
    print(f"  Exposition: {expositions[i]}")
    print(f"  Glazing_ratio: {df_building_info.loc[target_fid, 'GlazingRatio']}")
    print(f"  Building_age: {df_building_info.loc[target_fid, 'BuildingAge']}")
df_results = pd.DataFrame(results, columns=['TARGET_FID', 'Area', 'Width', 'Length', 'Exposition', 'GlazingRatio', 'BuildingAge'])

# Apply binning
df_results['Area_bin'] = bin_values(df_results['Area'])
df_results['Width_bin'] = bin_values(df_results['Width'])
df_results['Length_bin'] = bin_values(df_results['Length'])
df_results['GlazingRatio_bin'] = bin_values(df_results['GlazingRatio'])
df_results['BuildingAge_bin'] = bin_values(df_results['BuildingAge'])

# One-hot encoding for binned continuous variables and categorical variables
df_results = pd.get_dummies(df_results, columns=['Exposition', 'Area_bin', 'Width_bin', 'Length_bin', 'GlazingRatio_bin', 'BuildingAge_bin'])

# Print the DataFrame with one-hot encoded data
print(df_results.head())
df_results.drop(columns=['Area', 'Width', 'Length', 'GlazingRatio', 'BuildingAge'], inplace=True)
# Optionally, save the DataFrame to a CSV file
df_results.to_csv('Dataset/Processed_BuildingData.csv', index=False)