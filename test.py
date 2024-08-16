
import matplotlib.pyplot as plt
import csv
import os

import logging.config
import cesarp.common
import numpy as np

from shapely.geometry import Polygon

def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))

def find_longest_edge(polygon):
    coords = np.array(polygon.exterior.coords)
    max_length = 0
    longest_edge = None

    # Loop through each edge
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        edge_vector = end - start
        length = np.linalg.norm(edge_vector)

        if length > max_length:
            max_length = length
            longest_edge = (start, end)

    return longest_edge


def compute_orientation_from_edge(longest_edge):
    start, end = longest_edge
    edge_vector = np.array(end) - np.array(start)

    # Calculate the perpendicular direction
    perpendicular_direction = np.array([-edge_vector[1], edge_vector[0]])

    # Normalize direction vector
    norm = np.linalg.norm(perpendicular_direction)
    perpendicular_direction /= norm

    # Calculate the angle of orientation with respect to the x-axis
    angle = np.arctan2(perpendicular_direction[1], perpendicular_direction[0])
    angle_degrees = np.degrees(angle) % 360

    # Determine the closest compass direction
    if (angle_degrees >= 45 and angle_degrees < 135) or (angle_degrees >= 225 and angle_degrees < 315):
        orientation_category = 'N-S' if angle_degrees < 180 else 'S-N'
    elif (angle_degrees >= 135 and angle_degrees < 225):
        orientation_category = 'E-W' if angle_degrees < 270 else 'W-E'
    elif (angle_degrees >= 315 or angle_degrees < 45):
        orientation_category = 'NE-SW'
    else:
        orientation_category = 'NW-SE'

    return angle_degrees, orientation_category


# Define the polygon (e.g., building footprint)
# logging.config.fileConfig(
#         __abs_path("logging.conf")
#     )
# site_vertices_path = __abs_path("./data")
# input_csv = '/home/fazel/Documents/cesarp/cesarp_examples/random_idf_generator/data/Random_SiteVertices.csv'
# building_data = {}
# with open(input_csv, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)
#     for row in reader:
#         building_id = int(row[0])
#         x_coord = float(row[1])
#         y_coord = float(row[2])
#         z_coord = float(row[3])  # Assuming we need z for some reason
#         if building_id not in building_data:
#             building_data[building_id] = []
#         building_data[building_id].append((building_id, x_coord, y_coord, z_coord))
#
# # building_id = building_data[0]
# x_coords = [float(row[1]) for row in building_data]
# y_coords = [float(row[2]) for row in building_data]
coords = [(51.1307864481076, 38.4439874862758), (40.9859651744464, 57.3189670973629), (48.8692135518924, 61.5560125137242), (59.0140348255537, 42.6810329026371), (51.1307864481076, 38.4439874862758)]
coords = [(0, 0), (2.1, 0.3), (2, 1.4), (1.7, 1), (1.8, 3.3), (-0.2, 2), (0, 0)]
polygon = Polygon(coords)

# Find the longest edge and compute orientation
longest_edge = find_longest_edge(polygon)
orientation_angle, orientation_category = compute_orientation_from_edge(longest_edge)

print(f"Longest edge: {longest_edge}")
print(f"Orientation angle (degrees): {orientation_angle}")
print(f"Orientation category: {orientation_category}")

# Optional: Plotting the result
fig, ax = plt.subplots()
x, y = polygon.exterior.xy
ax.plot(x, y, 'b-', label='Polygon')

# Draw the longest edge
start, end = longest_edge
ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', label='Longest Edge')

# Draw the perpendicular direction (orientation direction)
centroid = np.mean(np.array(polygon.exterior.coords), axis=0)
orientation_direction = np.array([-end[1] + start[1], end[0] - start[0]])
ax.quiver(*centroid, *orientation_direction, angles='xy', scale_units='xy', scale=1, color='g', label='Orientation')

ax.set_aspect('equal')
plt.legend()
plt.show()
