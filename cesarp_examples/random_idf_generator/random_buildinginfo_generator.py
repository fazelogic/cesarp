
import pandas as pd
import numpy as np
import random
sample_size = 4
index = [i for i in range(1, sample_size+1)]
df = pd.DataFrame(index, columns =['ORIG_FID'])
df['SIA2024BuildingType'] = random.choice(["SFH", "MFH"])
df['BuildingAge'] = np.random.randint(1900, 2015, size=sample_size, dtype=int)
df.to_csv('Random_BuildingInformation.csv', index=False)


