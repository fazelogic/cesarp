import os
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging.config
import cesarp.common

def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))

# List to store data frames
dfs = []

# Iterate over the files
for i in range(1, 10):
    file_name = __abs_path(f"./results/example/EPF_inputs/fid_{i}.csv")
    df = pd.read_csv(file_name, header=None)  # Assuming no header in the CSV files
    dfs.append(df)

# Extracting and plotting the first column from each DataFrame
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

for i, df in enumerate(dfs):
    plt.plot(df.iloc[:, 2], label=f"idf_{i+1}")

plt.xlabel('Index')
plt.ylabel('First Column Values')
plt.title('First Column from idf_1.csv to idf_9.csv')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('first_column_plots2.png')

if __name__ == "__main__":
    # note: expected to be run in simple_example folder - otherwise adapt the path specifications as needed

    logging.config.fileConfig(
        __abs_path("logging.conf")
    )