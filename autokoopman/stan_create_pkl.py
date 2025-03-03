'''Stanley Bak
Create Pickle files for Aerobench training from eval.log''' 

import pickle
import argparse
import os
import numpy as np

from stan_manual_koopman import load_data

def main():
    '''main entry point'''

    parser = argparse.ArgumentParser(description="Script to process evaluation log file.")

    #eval_file_path="../output/expr_20240918_143039/PPO_DubinsRejoin_1c75e_00000_0_2024-09-18_14-30-42/eval/ckpt_200/eval.log"

    parser.add_argument(
        "--eval_log_path",
        required=True,
        type=str,
        help="Path to the evaluation log file (must exist)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.eval_log_path):
        print(f"Error: The file '{args.eval_log_path}' does not exist.")
        exit(1)

    cache_filename="dubins_data.pkl"

    # delete the cache file if it exists
    if os.path.exists(cache_filename):
        print(f'deleting {cache_filename}...')
        os.remove(cache_filename)

    # Set NumPy to raise an error on overflow
    np.seterr(over='raise', invalid='raise')
    np.set_printoptions(suppress=True) # no scientific notation

    print(f"loading data from {args.eval_log_path}...")
    load_data(args.eval_log_path)

if __name__ == '__main__':
    main()