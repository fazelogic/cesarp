import os
import csv
import pandas as pd
from multiprocessing import Pool
import cesarp.common
import logging.config


def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))


def count_columns(filename):
    """

    :param filename:
    :return:
    """
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        # Assuming the first row gives the column headers
        first_row = next(reader, None)
        if first_row:
            return len(first_row)
        else:
            return 0


def find_csv_files(folder):
    """

    :param folder:
    :return:
    """
    csv_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def process_csv(filename):
    """

    :param filename:
    :return:
    """
    # Get filename without extension
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    # Count columns
    num_columns = count_columns(filename)
    return base_filename, num_columns


if __name__ == '__main__':
    logging.config.fileConfig(
        __abs_path("logging.conf")
    )
    csv_files_path = __abs_path("./results/example/EPF_inputs")
    output_dir = __abs_path("./results/example")
    csv_files_list = find_csv_files(csv_files_path)
    with Pool() as pool:
        results = pool.map(process_csv, csv_files_list)

    # for multiprocessing turned off, uncomment below:
    # results = []
    # for filename in csv_files_list:
    #     results.append(process_csv(filename))

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=['File', 'Column Count'])
    print(df.iloc[:, 1].max())

    # Save DataFrame to CSV in output directory
    output_csv_path = os.path.join(output_dir, 'csv_column_counts.csv')
    df.to_csv(output_csv_path, index=False)
