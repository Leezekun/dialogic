
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
    # parser.add_argument('--raw_data_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-dialogic-processed/2_shot_augment_x2_dst_turn_info_train_ratio_0.01.json", help='the path that stores the error test cases in dst.')
    parser.add_argument('--raw_data_path', type=str, default="./simulation_result23/small/few_shot_0.05/combine0.2_2_shot_augment_dialog_turn_info_train_ratio_0.05_simulation_result.json", help='the path that stores the error test cases in dst.')
    parser.add_argument('--save_training_data_path', type=str, default="../../../simulated_dialogues/few_shot_0.05/", help='the path that stores the dialog demos.')

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

    """
    Start processing the data
    """
    processed_dialogs = []

    for aug_dialog in aug_dialogs:
        processed_dialog = {}
        processed_dialog["dial_id"] = aug_dialog["dial_id"]
        processed_dialog["goal"] = aug_dialog["goal"]

        dialog_turns = aug_dialog['turns']
        processed_dialog_turns = []
        for turn in dialog_turns:
            processed_turn = {}
            processed_turn['dial_id'] = turn['dial_id']
            processed_turn['turn_num'] = turn['turn_num']
            processed_turn['user'] = turn['user']
            processed_turn['resp'] = turn['resp']
            processed_turn['bspn'] = turn['bspn']
            processed_turn['aspn'] = turn['aspn']
            processed_turn['db'] = turn['db']
            processed_dialog_turns.append(processed_turn)
    
        processed_dialog['turns'] = processed_dialog_turns
        processed_dialog['prompt'] = aug_dialog['prompt']

        processed_dialogs.append(processed_dialog)

    print(f"{len(processed_dialogs)} dialogs in total. Finished transferring augmented dialogs in training data format......")

    print("Start saving augmented dialogs in training data format......")
    # save the augmented dialog turn info
    if not os.path.exists(args.save_training_data_path):
        os.makedirs(args.save_training_data_path, exist_ok=True)
    
    save_training_data_path = os.path.join(args.save_training_data_path, one_dev_str + ".json")
    f = open(save_training_data_path, "w")
    json.dump(processed_dialogs, f)
    f.close()

    """
    Start processing the data
    """
    processed_dialogs = []

    for aug_dialog in aug_dialogs:
        processed_dialog = copy.deepcopy(aug_dialog)

        dialog_turns = aug_dialog['turns']
        processed_dialog_turns = []

        for turn in dialog_turns:
            processed_turn = {}
            for k, v in turn.items():
                if "_gen" not in k:
                    processed_turn[k] = v
            processed_dialog_turns.append(processed_turn)
    
        processed_dialog['turns'] = processed_dialog_turns
        processed_dialogs.append(processed_dialog)

    print(f"{len(processed_dialogs)} dialogs in total. Finished transferring augmented dialogs in training data format......")

    print("Start saving augmented dialogs in training data format......")
    
    f = open(args.raw_data_path, "w")
    json.dump(processed_dialogs, f)
    f.close()


