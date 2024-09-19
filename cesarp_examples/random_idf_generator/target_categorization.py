
import csv
import os
import math
import multiprocessing
import logging.config

import pandas as pd

import cesarp.common
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate


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
    if angle_degrees < 22.5:
        orientation_category = 1
    elif angle_degrees < 67.5:
        orientation_category = 2
    elif angle_degrees < 112.5:
        orientation_category = 3
    elif angle_degrees < 157.5:
        orientation_category = 4
    elif angle_degrees < 202.5:
        orientation_category = 1
    elif angle_degrees < 247.5:
        orientation_category = 2
    elif angle_degrees < 292.5:
        orientation_category = 3
    elif angle_degrees < 337.5:
        orientation_category = 4
    else:
        orientation_category = 1

    return angle_degrees, orientation_category


# def digitize_area(area):
#     area = np.array(area)
#     num_bins = 7
#     # Calculate bin edges
#     bin_edges = np.linspace(area.min(), area.max(), num_bins + 1)
#     # Assign each value to a bin
#     area_bin = np.digitize(area, bins=bin_edges)
#     # Create numpy array with actual value and assigned bin number
#     # area_bin = np.column_stack((area, bin_indices))
#     return area_bin


def digitize(original_array, num_bins):
    # Calculate the bin edges using quantiles to ensure equal population in each bin
    bin_edges = np.quantile(original_array, np.linspace(0, 1, num_bins + 1))

    # In case you need the bin edges, uncomment the two lines below:
    # print("elongation")
    # print(bin_edges)

    # Assign each element in the array to a bin
    bins = np.digitize(original_array, bin_edges[1:], right=True)  # Start from the second edge to avoid bin 0

    # elongation = np.array(elongation)
    # num_bins = 7
    # # Calculate bin edges
    # bin_edges = np.linspace(elongation.min(), elongation.max(), num_bins + 1)
    # # Assign each value to a bin
    # elongation_bin = np.digitize(elongation, bins=bin_edges)
    # # Create numpy array with actual value and assigned bin number
    # # elongation_bin = np.column_stack((elongation, bin_indices))
    return bins


def one_hot_encode(input_list):
    unique_values, indices = np.unique(input_list, return_inverse=True)
    one_hot_matrix = np.eye(len(unique_values))[indices]
    return one_hot_matrix

# Function to determine orientation
def calculate_orientation(x_coords, y_coords):
    poly = Polygon(zip(x_coords, y_coords))

    # Find the longest edge and compute orientation
    longest_edge = find_longest_edge(poly)
    orientation_angle, orientation_category = compute_orientation_from_edge(longest_edge)
    return orientation_category


# Function to calculate area
def calculate_area(x_coords, y_coords):
    poly = Polygon(zip(x_coords, y_coords))
    return poly.area


# Function to calculate elongation
def calculate_elongation(x_coords, y_coords):
    poly = Polygon(zip(x_coords, y_coords))
    assert tuple(poly.exterior.coords) == tuple(zip(x_coords, y_coords))
    # Step 1: Calculate the convex hull of the polygon
    # coordinates = list(poly.exterior.coords)
    # hull = ConvexHull(coordinates)
    # hull_points = np.array([coordinates[i] for i in hull.vertices])

    # Step 2: Find the minimum bounding box by rotating the polygon and checking bounds
    min_area = float('inf')
    min_bbox = None
    # min_angle = 0

    for angle in np.arange(0, 180, 1):
        rotated_polygon = rotate(poly, angle, use_radians=False)
        bounds = rotated_polygon.bounds  # returns (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area = width * height

        if area < min_area:
            min_area = area
            min_bbox = bounds
            # min_angle = angle

    # Step 3: Calculate the length (max of width and height) and width (min of width and height)
    bbox_width = min_bbox[2] - min_bbox[0]
    bbox_length = min_bbox[3] - min_bbox[1]

    # Ensure length is the longer side and width is the shorter side
    if bbox_width > bbox_length:
        bbox_width, bbox_length = bbox_length, bbox_width

    elongation_ratio = bbox_width / bbox_length
    return elongation_ratio


# Function to process each building
def process_building(building_data):
    building_id = building_data[0][0]
    x_coords = [float(row[1]) for row in building_data]
    y_coords = [float(row[2]) for row in building_data]
    # z_coords = [float(row[3]) for row in building_data]

    # calculate orientation bin
    orientation = calculate_orientation(x_coords, y_coords)

    # calculate area bin
    area = calculate_area(x_coords, y_coords)

    # calculate elongation bin
    elongation = calculate_elongation(x_coords, y_coords)

    return building_id, orientation, area, elongation


# Python program to get transpose
# elements of two dimension list
def transpose(l1, l2):
    # star operator will first
    # unpack the values of 2D list
    # and then zip function will
    # pack them again in opposite manner
    l2 = list(map(list, zip(*l1)))
    return l2


# Function to write results to CSV
def write_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Building ID', 'Orientation', 'Area', 'Elongation'])
        writer.writerows(results)


def main(input_csv, misc_csv, output_csv):
    # Read input CSV file
    building_data = {}
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            building_id = int(row[0])
            x_coord = float(row[1])
            y_coord = float(row[2])
            z_coord = float(row[3])  # Assuming we need z for some reason
            if building_id not in building_data:
                building_data[building_id] = []
            building_data[building_id].append((building_id, x_coord, y_coord, z_coord))

    # Use multiprocessing to process each building
    pool = multiprocessing.Pool()
    results = pool.map(process_building, building_data.values())
    pool.close()
    pool.join()

    # For no multiprocessing uncomment below
    # results = []
    # for building_id, data in building_data.items():
    #     result = process_building(data)
    #     results.append(result)

    # orientation
    cats = pd.DataFrame(one_hot_encode(list(zip(*results))[1]), columns=['or1', 'or2', 'or3', 'or4'])
    # area
    area = [row[2] for row in results]
    area_bin = digitize(area, num_bins=8)
    area_bin = one_hot_encode(area_bin)
    area_bin = pd.DataFrame(area_bin, columns=['ar1', 'ar2', 'ar3', 'ar4', 'ar5', 'ar6', 'ar7', 'ar8'])
    cats = pd.concat([cats, area_bin], axis=1)

    # elongation
    elongation = [row[3] for row in results]
    elongation_bin = digitize(elongation, num_bins=8)
    elongation_bin = one_hot_encode(elongation_bin)
    elongation_bin = pd.DataFrame(elongation_bin, columns=['el1', 'el2', 'el3', 'el4', 'el5', 'el6', 'el7', 'el8'])
    cats = pd.concat([cats, elongation_bin], axis=1)

    # WWR
    temp_csv = pd.read_csv(misc_csv, sep=',')
    wwr = temp_csv["GlazingRatio"]
    digi_wwr = digitize(wwr.to_numpy(), num_bins=5)
    digi_wwr = pd.DataFrame(one_hot_encode(np.transpose(digi_wwr)), columns=['wwr1', 'wwr2', 'wwr3', 'wwr4', 'wwr5'])
    cats = pd.concat([cats, digi_wwr], axis=1)

    # YoC
    temp_csv = pd.read_csv(misc_csv, sep=',')
    yoc = temp_csv["BuildingAge"]
    digi_yoc = digitize(yoc.to_numpy(), num_bins=5)
    digi_yoc = pd.DataFrame(one_hot_encode(np.transpose(digi_yoc)), columns=['yoc1', 'yoc2', 'yoc3', 'yoc4', 'yoc5'])
    cats = pd.concat([cats, digi_yoc], axis=1)

    # Write results to CSV
    cats.to_csv(output_csv, index=False)


if __name__ == '__main__':
    logging.config.fileConfig(
        __abs_path("logging.conf")
    )
    site_vertices_path = __abs_path("./data")
    output_dir = __abs_path("./results/example")
    input_csv_path = os.path.join(site_vertices_path, 'Random_SiteVertices.csv')
    misc_csv_path = os.path.join(site_vertices_path, 'Random_BuildingInformation.csv')
    output_csv_path = os.path.join(output_dir, 'Categorized_Targets.csv')
    main(input_csv_path, misc_csv_path, output_csv_path)
