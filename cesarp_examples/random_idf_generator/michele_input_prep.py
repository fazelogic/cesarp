
import os
import pandas as pd
import numpy as np
import multiprocessing as mp



def get_max_columns(file_list):
    max_columns = 0
    for file in file_list:
        df = pd.read_csv(file)
        max_columns = max(max_columns, df.shape[1])
    return max_columns

def pad_and_save(file, max_columns):
    df = pd.read_csv(file)
    padded_array = np.zeros((df.shape[0], max_columns))
    mask_array = np.zeros((df.shape[0], max_columns))

    padded_array[:, :df.shape[1]] = df.values
    mask_array[:, :df.shape[1]] = 1  # Mask original data with 1s

    # Save the padded array and mask
    np.save(os.path.join(numpy_directory, os.path.basename(file).replace('.csv', '_data.npy')), padded_array)
    np.save(os.path.join(numpy_directory, os.path.basename(file).replace('.csv', '_mask.npy')), mask_array)



######################################################







#######################################################

def normalize_data(data_file, global_mins, global_maxs):
    mask_file = data_file.replace('_data.npy', '_mask.npy')
    data = np.load(data_file)
    mask = np.load(mask_file)

    normalized_data = np.zeros_like(data, dtype=np.float32)

    for col in range(data.shape[1]):
        valid_data = data[mask[:, col] == 1, col]

        if len(valid_data) > 0:
            min_val = global_mins[col]
            max_val = global_maxs[col]

            normalized_data[mask[:, col] == 1, col] = (valid_data - min_val) / (max_val - min_val)

    np.save(os.path.join(numpy_directory, os.path.basename(data_file).replace('_data.npy', '_data_norm.npy')), normalized_data)


    ################################################################

#check that all the files are in the same range

def calculate_min_max(data_file):
    mask_file = data_file.replace('_data_norm.npy', '_mask.npy')
    data = np.load(data_file)
    mask = np.load(mask_file)

    num_columns = data.shape[1]
    mins = np.full(num_columns, np.inf)  # Initialize with large values
    maxs = np.full(num_columns, -np.inf)  # Initialize with small values

    for col in range(num_columns):
        valid_data = data[mask[:, col] == 1, col]
        if len(valid_data) > 0:
            mins[col] = np.min(valid_data)
            maxs[col] = np.max(valid_data)

    return (mins, maxs)



##################

if __name__ == '__main__':
    # Directory containing CSV files
    csv_directory = '/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/EPF_inputs'
    # Directory to save numpy files
    numpy_directory = '/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/npy_inputs'

    os.makedirs(numpy_directory, exist_ok=True)
    print("0")

    # Get list of CSV files
    csv_files = [os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if f.endswith('.csv')]
    print("1")

    # Determine the maximum number of columns across all files
    # with mp.Pool(mp.cpu_count()) as pool:
    #     max_columns = pool.map(get_max_columns, csv_files)
    max_columns = 9  # print get_max_columns(csv_files)

    print("2")

    # Use multiprocessing to pad and save numpy files
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(pad_and_save, [(file, max_columns) for file in csv_files])
    pool.close()
    pool.join()

    print(f"Processed {len(csv_files)} files and saved them as numpy arrays along with their masks.")



    # Get list of all data files
    data_files = [os.path.join(numpy_directory, f) for f in os.listdir(numpy_directory) if f.endswith('_data.npy')]
    csv_files = [os.path.join(csv_directory, f) for f in os.listdir(csv_directory) if f.endswith('_data.npy')]
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(calculate_min_max, data_files)

    # Aggregate mins and maxs across all files
    all_mins = np.min([result[0] for result in results if result[0] is not None], axis=0)
    all_maxs = np.max([result[1] for result in results if result[1] is not None], axis=0)

    # Save global mins and maxs
    np.save('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/global_mins.npy', all_mins)
    np.save('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/global_maxs.npy', all_maxs)

    print("3")

    global_mins = np.load('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/global_mins.npy')
    global_maxs = np.load('/home/fazel/PycharmProjects/cesarp/cesarp_examples/random_idf_generator/results/example/global_maxs.npy')

    print(global_mins, global_maxs)

    data_files = [os.path.join(numpy_directory, f) for f in os.listdir(numpy_directory) if f.endswith('_data.npy')]

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(normalize_data, [(data_file, global_mins, global_maxs) for data_file in data_files])

        # Get list of all data files
        data_files = [os.path.join(numpy_directory, f) for f in os.listdir(numpy_directory) if
                      f.endswith('_data_norm.npy')]

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(calculate_min_max, data_files)

        # Aggregate mins and maxs across all files
        all_mins = np.min([result[0] for result in results if result[0] is not None], axis=0)
        all_maxs = np.max([result[1] for result in results if result[1] is not None], axis=0)

        # Save global mins and maxs
        print(all_mins)
        print(all_maxs)