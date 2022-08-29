
import progressbar
import argparse
import logging
import time
import json
import random
import re
import copy
import os

def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_type', type=str, help='the type of training data, choice: [E2E, DST].')

    # parser.add_argument('--raw_data_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-dialogic-processed/2_shot_augment_x2_dst_turn_info_train_ratio_0.01.json", help='the path that stores the error test cases in dst.')
    parser.add_argument('--raw_data_path', type=str, default="./simulation_result23/small/few_shot_0.01/combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.01_simulation_result.json", help='the path that stores the error test cases in dst.')
    parser.add_argument('--save_training_data_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-fine-processed/", help='the path that stores the dialog demos.')

    return parser.parse_args()

    
import argparse
if __name__ == '__main__':
    args = parse_config()

    # load the inference results (usually error ones in dev)
    print("Start loading inference results......")
    assert args.raw_data_path is not None
    one_dev_str = args.raw_data_path.split("/")[-1].split(".json")[0].strip()
    f = open(args.raw_data_path, "r")
    aug_dialogs = json.load(f) # dict, dict[dial_id] is dialog_dict, dialog_dict[turn_id] is a turn_dict
    f.close()
        
    if args.data_type == "E2E":
        data_type = "E2E"
    elif args.data_type == "DST":
        data_type = "DST"
    else:
        raise Exception("Wrong data type!")

    """
    Start processing the data
    """
    training_data = []

    if data_type == "E2E":
        for aug_dialog in aug_dialogs:
            processed_dialog = []

            dial_id = aug_dialog['dial_id']
            dialog_turns = aug_dialog['turns']
            dialog_goal = aug_dialog['goal']
            dialog_prompt = aug_dialog['prompt']
        
            for turn in dialog_turns:
                processed_turn = {}
                processed_turn['dial_id'] = turn['dial_id']
                processed_turn['user'] = "<sos_u> " + turn['user'] + " <eos_u>"
                processed_turn['usdx'] = "<sos_u> " + turn['usdx'] + " <eos_u>"
                processed_turn['resp'] = "<sos_r> " + turn['resp'] + " <eos_r>"
                processed_turn['nodelx_resp'] = "<sos_r> " + turn['resp'] + " <eos_r>"
                processed_turn['bspn'] = "<sos_b> " + turn['bspn'] + " <eos_b>"
                processed_turn['bsdx'] = "<sos_b> " + turn['bsdx'] + " <eos_b>"
                processed_turn['aspn'] = "<sos_a> " + turn['aspn'] + " <eos_a>"
                processed_turn['dspn'] = "<sos_d> " + turn['aspn'] + " <eos_d>"
                processed_turn['pointer'] = turn['pointer']
                processed_turn['turn_domain'] = turn['turn_domain']
                processed_turn['turn_num'] = turn['turn_num']
                processed_turn['db'] = "<sos_db> " + turn['db'] + " <eos_db>"
                processed_turn['bspn_reform'] = "<sos_b> " + turn['bspn_reform'] + " <eos_b>"
                processed_turn['bsdx_reform'] = "<sos_b> " + turn['bsdx_reform'] + " <eos_b>"
                processed_turn['aspn_reform'] = "<sos_a> " + turn['aspn_reform'] + " <eos_a>"
                
                processed_dialog.append(processed_turn)
        
            training_data.append(processed_dialog)

        print(f"{len(training_data)} dialogs in total. Finished transferring augmented dialogs in training data format......")

    elif data_type == "DST":
        for aug_dial_id, aug_dialog in aug_dialogs.items():
            processed_dialog = []

            for turn_id, turn in aug_dialog.items():
                processed_turn = {}
                processed_turn['dial_id'] = turn['dial_id']
                processed_turn['user'] = "<sos_u> " + turn['user'] + " <eos_u>"
                processed_turn['usdx'] = "<sos_u> " + turn['user'] + " <eos_u>"
                processed_turn['resp'] = "<sos_r> " + turn['resp'] + " <eos_r>"
                processed_turn['nodelx_resp'] = "<sos_r> " + turn['nodelx_resp'] + " <eos_r>"
                processed_turn['bspn'] = "<sos_b> " + turn['bspn'] + " <eos_b>"
                processed_turn['bsdx'] = "<sos_b> " + turn['bsdx'] + " <eos_b>"
                processed_turn['aspn'] = "<sos_a> " + turn['aspn'] + " <eos_a>"
                processed_turn['dspn'] = "<sos_d> " + turn['aspn'] + " <eos_d>"
                processed_turn['pointer'] = turn['pointer']
                processed_turn['turn_domain'] = turn['turn_domain']
                processed_turn['turn_num'] = turn['turn_num']
                processed_turn['db'] = "<sos_db> " + turn['db'] + " <eos_db>"
                processed_turn['bspn_reform'] = "<sos_b> " + turn['bspn_reform'] + " <eos_b>"
                processed_turn['bsdx_reform'] = "<sos_b> " + turn['bsdx_reform'] + " <eos_b>"
                processed_turn['aspn_reform'] = "<sos_a> " + turn['aspn_reform'] + " <eos_a>"
                
                augment_turns = turn['augment_turns']
                processed_augment_turns = []
                for augment_turn in augment_turns:
                    processed_augment_turn = {}
                    processed_augment_turn['user'] = "<sos_u> " + augment_turn['user'] + " <eos_u>"
                    processed_augment_turn['usdx'] = "<sos_u> " + augment_turn['user'] + " <eos_u>"
                    processed_augment_turn['bspn'] = "<sos_b> " + augment_turn['bspn'] + " <eos_b>"
                    processed_augment_turn['bsdx'] = "<sos_b> " + augment_turn['bsdx'] + " <eos_b>"
                    processed_augment_turn['bspn_reform'] = "<sos_b> " + augment_turn['bspn_reform'] + " <eos_b>"
                    processed_augment_turn['bsdx_reform'] = "<sos_b> " + augment_turn['bsdx_reform'] + " <eos_b>"
                    processed_augment_turns.append(processed_augment_turn)
                
                processed_turn['augment_turns'] = processed_augment_turns

                processed_dialog.append(processed_turn)
        
            training_data.append(processed_dialog)

        print(f"{len(training_data)} dialogs in total. Finished transferring augmented dialogs in training data format......")

    print("Start saving augmented dialogs in training data format......")
    # save the augmented dialog turn info
    assert args.save_training_data_path is not None
    save_training_data_path = os.path.join(args.save_training_data_path, "multi-woz-fine-processed-train-" + one_dev_str + ".json")
    f = open(save_training_data_path, "w")
    json.dump(training_data, f)
    f.close()


