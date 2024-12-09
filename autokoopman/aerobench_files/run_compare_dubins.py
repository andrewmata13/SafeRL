'''load rollouts from safeRL and compare to f-16 recreations with same inputs

Get a predictive model for online RTA
'''

import numpy as np

from compare_util import load_data_and_sim_f16, eval_predictor_accuracy, DubinsPredictor, make_linear_predictor, plot_prediction_vs_dubins

def main():
    '''main function'''

    np.set_printoptions(suppress=True, precision=3)

    all_f16_res_dicts = load_data_and_sim_f16(single_index=None)

    MAX_STEPS_TO_PREDICT = 19
    predictor = make_linear_predictor(all_f16_res_dicts, MAX_STEPS_TO_PREDICT)

    for num_steps_to_predict in [MAX_STEPS_TO_PREDICT]: #range(1, 20):
        

        max_error_vector = eval_predictor_accuracy(all_f16_res_dicts, predictor, num_steps_to_predict=num_steps_to_predict)
        print(f"{num_steps_to_predict=}, {max_error_vector=}")

        plot_prediction_vs_dubins(all_f16_res_dicts, predictor, num_steps_to_predict=num_steps_to_predict)

    #compare_dubins_data() # create data file from f-16 recreations of dubins paths

if __name__ == '__main__':
    main()
