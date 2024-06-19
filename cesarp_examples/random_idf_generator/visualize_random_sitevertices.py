import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon


def plot_buildings(buildings_df):
    """

    :param buildings_df: a dataframe of building vertices
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get unique building IDs
    unique_buildings = buildings_df['TARGET_FID'].unique()

    for building_id in unique_buildings:
        # Filter rows for each building
        building_vertices = buildings_df[buildings_df['TARGET_FID'] == building_id]

        # Extract vertices for the Polygon
        vertices = building_vertices[['POINT_X', 'POINT_Y']].values
        polygon = Polygon(vertices)
        # print(polygon.area)

        # Create a patch from the Polygon
        patch = patches.Polygon(xy=list(polygon.exterior.coords), closed=True, fill=True, edgecolor='black',
                                facecolor='grey', linewidth=2, alpha=0.5)
        ax.add_patch(patch)

    # Set the limits of the plot to the limits of the data
    ax.set_xlim(buildings_df['POINT_X'].min(), buildings_df['POINT_X'].max())
    ax.set_ylim(buildings_df['POINT_Y'].min(), buildings_df['POINT_Y'].max())
    ax.set_aspect('equal', 'box')

    plt.title('Building Layout')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()


plt.switch_backend('TkAgg')
# Assuming 'centered_buildings_df' is already created and available
plot_buildings(pd.read_csv('data/Random_SiteVertices.csv'))
