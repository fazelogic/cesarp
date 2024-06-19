# coding=utf-8

import pandas as pd
import numpy as np
import random


def generate_simple_building_info(sample_size):
    """

    :param sample_size: number of buildings to be generated
    """
    index = [i for i in range(1, sample_size+1)]
    building_info_df = pd.DataFrame(index, columns=['ORIG_FID'])
    building_info_df['SIA2024BuildingType'] = random.choice(["SFH", "MFH"])
    building_info_df['BuildingAge'] = np.random.randint(1900, 2020, size=sample_size, dtype=int)
    building_info_df.to_csv('data/Random_BuildingInformation.csv', index=False)

