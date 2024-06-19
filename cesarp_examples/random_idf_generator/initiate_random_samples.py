# coding=utf-8

import numpy as np
from cesarp_examples.random_idf_generator.random_sitevertices_generator import generate_centered_buildings_in_grid
from cesarp_examples.random_idf_generator.random_buildinginfo_generator import generate_simple_building_info

number_of_buildings = 4
preferred_min_width = 5
preferred_max_width = 40
preferred_min_height = 3
preferred_max_height = 19


grid_root = int(np.floor(np.sqrt(number_of_buildings)))
sample_size = int(np.square(grid_root))
block_size = int(np.floor(preferred_max_width * 2.5))
min_building_size = np.floor(preferred_min_width)
max_building_size = np.floor(preferred_max_width)
min_building_height = np.floor(preferred_min_height)
max_building_height = np.floor(preferred_max_height)

centered_buildings_df = generate_centered_buildings_in_grid(grid_root, block_size, min_building_size,
                                                            max_building_size, min_building_height,
                                                            max_building_height)
generate_simple_building_info(sample_size)
