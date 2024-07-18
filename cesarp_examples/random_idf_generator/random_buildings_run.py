#
# Copyright (c) 2021, Empa, Leonie Fierz
#
# This file is part of CESAR-P - Combined Energy Simulation And Retrofit written in Python
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Contact: https://www.empa.ch/web/s313
#
import logging.config
import logging
import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

import cesarp.common
import cesarp.common.config_loader
from cesarp.manager.SimulationManager import SimulationManager
from cesarp.eplus_adapter.eplus_eso_results_handling import (RES_KEY_EL_DEMAND, RES_KEY_HEATING_DEMAND,
                                                             RES_KEY_INDOOR_TEMPERATURE)
from cesarp.eplus_adapter.idf_strings import ResultsFrequency
from initiate_random_samples import initiate_random_samples


def __abs_path(path):
    return cesarp.common.abs_path(path, os.path.abspath(__file__))


def create_directory_if_not_exists(directory):
    """
    create a directory if it doesn't exist
    :param directory:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_fid(fids):
    """

    :param fids:
    """
    sub_results = result_series_frame[result_series_frame['fid'] == fids]
    sub_results = pd.DataFrame(sub_results.iloc[:, 2].values.reshape(-1, 8760), columns=np.arange(1, 8761)).transpose()
    sub_results.iloc[:, :2] = sub_results.iloc[:, :2] / 3600000
    sub_results.to_csv(__abs_path(save_dir) / Path(f"fid_{fids}.csv"), header=False, index=False)


def move_files(truncated_idf):
    """

    :param truncated_idf:
    """
    src_directory = os.path.join(truncated_dir, truncated_idf)
    dst_directory = os.path.join(save_dir, truncated_idf)
    shutil.move(src_directory, dst_directory)


if __name__ == "__main__":
    # note: expected to be run in simple_example folder - otherwise adapt the path specifications as needed.
    # This logging config is only for the main process, workers log to separate log files which go into a folder,
    # configured in SimulationManager.

    logging.config.fileConfig(
        __abs_path("logging.conf")
    )

    main_config_path = __abs_path("random_buildings_config.yml")
    output_dir = __abs_path("./results/example")
    truncated_dir = __abs_path("./results/example/idfs/truncated_idfs")
    save_dir = __abs_path("./results/example/EPF_inputs")
    shutil.rmtree(output_dir, ignore_errors=True)
    create_directory_if_not_exists(truncated_dir)
    sample_size = 100000

    initiate_random_samples(sample_size)

    fids_to_use = None  # None or [1, 2, 3]  -> set to None to simulate all buildings
    sim_manager = SimulationManager(output_dir, main_config_path, cesarp.common.init_unit_registry(),
                                    fids_to_use=fids_to_use)
    sim_manager.run_all_steps()

    print("\n=========SIMULATIONS FINISHED ===========")
    # if you need different result parameters, you have to make sure that EnergyPlus reports them. Do so by using the
    # configuration parameters from eplus_adapter package, namely "OUTPUT_METER" and "OUTPUT_VARS",
    # see cesarp.eplus_adapter.default_config.yml. You can overwrite those parameters in your project config,
    # in this example that would be simple_main_config.yml.
    # Also make sure that the reporting frequency in the configuration and in the collect_custom_results() call match.
    create_directory_if_not_exists(save_dir)
    result_series_frame = sim_manager.save_custom_results(result_keys=[RES_KEY_HEATING_DEMAND, RES_KEY_EL_DEMAND,
                                                                       RES_KEY_INDOOR_TEMPERATURE],
                                                          results_frequency=ResultsFrequency.HOURLY,
                                                          save_folder_path=__abs_path(save_dir))
    print("\n=========EXTRACTING RESULTS FINISHED ===========")
    # you can post-process the results as you like, e.g. save to a file


    # num_fids = result_series_frame['fid'].max()+1
    with Pool(processes=cpu_count()-2) as pool:
        pool.map(move_files, os.listdir(truncated_dir))
    os.rmdir(truncated_dir)
    print("\n=========MOVING IDF FINISHED ===========")

    # zip_path = sim_manager.save_to_zip(main_script_path=__file__)
    print("\n=========PROCESS FINISHED ===========")
    print(f"You find the results in {save_dir}")
