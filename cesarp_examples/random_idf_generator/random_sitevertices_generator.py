import random
import pandas as pd
from shapely.geometry import box
from shapely.affinity import rotate, translate


def generate_centered_buildings_in_grid_optimized(grid_size, block_size, max_building_size):
    building_data = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the center of the block
            center_x = (i * block_size) + (block_size / 2)
            center_y = (j * block_size) + (block_size / 2)

            # Determine building dimensions and orientation
            width = random.uniform(20, max_building_size)  # minimum width is 20 for variety
            depth = random.uniform(20, max_building_size)
            angle = random.uniform(0, 360)  # rotation angle in degrees
            height = random.uniform(3, 12)  # height of each building

            # Create a rectangle centered at the origin, rotate, and then translate it to the block center
            rect = box(-width / 2, -depth / 2, width / 2, depth / 2)
            rotated_rect = rotate(rect, angle, origin='center', use_radians=False)
            final_rect = translate(rotated_rect, xoff=center_x, yoff=center_y)

            # Store the vertices in the list, including closing the loop by repeating the first vertex at the end
            vertices = list(final_rect.exterior.coords)
            for vx, vy in vertices:
                building_data.append({
                    "TARGET_FID": i * grid_size + j + 1,
                    "POINT_X": vx,
                    "POINT_Y": vy,
                    "HEIGHT": height
                })

    # Create DataFrame from list
    buildings_df = pd.DataFrame(building_data, columns=["TARGET_FID", "POINT_X", "POINT_Y", "HEIGHT"])
    return buildings_df


# Parameters
grid_size = 2  # 100x100 grid
block_size = 2  # Each block is 100m x 100m
max_building_size = 40  # Maximum size of the building width or depth

# Generate the buildings using the optimized approach
centered_buildings_df = generate_centered_buildings_in_grid_optimized(grid_size, block_size, max_building_size)

centered_buildings_df.to_csv('Random_SiteVertices.csv', index=False)
