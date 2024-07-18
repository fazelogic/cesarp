
import csv
import os
import math
import multiprocessing
import numpy as np
from functools import partial
from shapely.geometry import Polygon
import logging.config
import cesarp.common


def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))


def digitize_area(area):
    area = np.array(area)
    num_bins = 7
    # Calculate bin edges
    bin_edges = np.linspace(area.min(), area.max(), num_bins + 1)
    # Assign each value to a bin
    area_bin = np.digitize(area, bins=bin_edges)
    # Create numpy array with actual value and assigned bin number
    # area_bin = np.column_stack((area, bin_indices))
    return area_bin


def digitize_elongation(elongation):
    elongation = np.array(elongation)
    num_bins = 7
    # Calculate bin edges
    bin_edges = np.linspace(elongation.min(), elongation.max(), num_bins + 1)
    # Assign each value to a bin
    elongation_bin = np.digitize(elongation, bins=bin_edges)
    # Create numpy array with actual value and assigned bin number
    # elongation_bin = np.column_stack((elongation, bin_indices))
    return elongation_bin

# Function to determine orientation
def calculate_orientation(x, y):
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    # Determine orientation based on angle
    directions = [1, 2, 3]  #, "SE", "S", "SW", "W", "NW"]
    index = round(angle / 45) % 8
    return directions[index]


# Function to calculate area
def calculate_area(x_coords, y_coords):
    poly = Polygon(zip(x_coords, y_coords))
    return poly.area


# Function to calculate elongation
def calculate_elongation(x_coords, y_coords):
    poly = Polygon(zip(x_coords, y_coords))
    bbox = poly.bounds
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if height == 0:
        return float('inf')  # to handle very flat buildings
    return width / height


# Function to process each building
def process_building(building_data):
    building_id = building_data[0][0]
    x_coords = [float(row[1]) for row in building_data]
    y_coords = [float(row[2]) for row in building_data]
    z_coords = [float(row[3]) for row in building_data]

    # calculate orientation bin
    orientation = calculate_orientation(x_coords[0], y_coords[0])

    # calculate area bin
    area = calculate_area(x_coords, y_coords)

    # calculate elongation bin
    elongation = calculate_elongation(x_coords, y_coords)

    return building_id, orientation, area, elongation


# Function to write results to CSV
def write_to_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Building ID', 'Orientation', 'Area', 'Elongation'])
        writer.writerows(results)


def main(input_csv, output_csv):
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

    area = list(zip(*results))[2]
    area_bin = digitize_area(area)
    print(max(area_bin))
    results = [(x[0], x[1], area_bin, x[3]) for x, area_bin in zip(results, area_bin)]
    elongation = list(zip(*results))[3]
    elongation_bin = digitize_elongation(elongation)
    print(max(elongation_bin))
    results = [(x[0], x[1], x[2], elongation_bin) for x, elongation_bin in zip(results, elongation_bin)]

    # Write results to CSV
    write_to_csv(results, output_csv)


if __name__ == '__main__':
    logging.config.fileConfig(
        __abs_path("logging.conf")
    )
    site_vertices_path = __abs_path("./data")
    output_dir = __abs_path("./results/example")
    intput_csv_path = os.path.join(site_vertices_path, 'Random_SiteVertices.csv')
    output_csv_path = os.path.join(output_dir, 'Categorized_Targets.csv')
    main(intput_csv_path, output_csv_path)
