
from ast import arg
import enum
from pickle import NONE
import progressbar
import argparse
import logging
import time
import json
import random
import re
import copy
import os
import numpy as np

from dialogic_utils import *


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration        
    parser.add_argument('--data_path_prefix', type=str, default='../data/multiwoz/data', help='The path where the data stores.')
    parser.add_argument('--data_version', type=str, default='2.3', help='The version of used multiwoz data, 2.0, 2.1, 2.3, 2.4')

    parser.add_argument('--model_name', type=str, default='t5-small', 
        help="the model type of t5, t5-small, t5-base, or t5-large.")

    parser.add_argument('--shuffle_mode', type=str, default='unshuffle', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--cascaded', type=str, default='False', 
        help="True or False, whether includes action when generating response.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    
    parser.add_argument('--train_data_ratio', type=float, default=0.01, help='the ratio of training data used for training the model')
    parser.add_argument('--aug_train_data_file', type=str, default="", help='the augmented data file path')
    parser.add_argument('--aug_train_data_num', type=int, default=-1, help='the number of used augmented data')

    parser.add_argument('--save_data_path_prefix', type=str, default='../../simpletod/resources_e2e_2.3_0.01_augx1/multi-woz', help='The path where the data stores.')
    # parser.add_argument('--save_data_path_prefix', type=str, default='../../MinTL/generated_data/e2e_2.3_0.01_augx1/', help='The path where the data stores.')
    # parser.add_argument('--save_data_path_prefix', type=str, help='The path where the data stores.')

    return parser.parse_args()


import argparse
if __name__ == '__main__':
    args = parse_config()
    
    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        from config import Config

    elif args.data_version == "2.1":
        from config21 import Config

    elif args.data_version == "2.3":
        from config23 import Config

    elif args.data_version == "2.4":
        from config24 import Config

    else:
        raise Exception("Wrong MultiWOZ version!")

    cfg = Config(args.data_path_prefix)
    assert args.model_name.startswith('t5')
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    if args.use_db_as_input == 'True':
        use_db_as_input = True
    elif args.use_db_as_input == 'False':
        use_db_as_input = False
    else:
        raise Exception('Wrong Use DB Mode!!!')

    if args.cascaded == 'True':
        cascaded = True
    elif args.cascaded == 'False':
        cascaded = False
    else:
        raise Exception('Wrong Use Cascaded Mode!!!')

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    dialogs_analysis = {}

    # load dialogs
    if args.aug_train_data_file:
        data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio, aug_train_data_file=args.aug_train_data_file, aug_train_data_num=args.aug_train_data_num)
    else:
        data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)

    print("Start converting raw multiwoz data for simpletod......")
    train_dials, dev_dials, test_dials, train_dial_id_list, dev_dial_id_list, test_dial_id_list = data.convert_to_raw_data()

    print("Start saving raw multiwoz data......")
    # save the dialog turn info
    if not os.path.exists(args.save_data_path_prefix):
        os.makedirs(args.save_data_path_prefix)

    if "human_evaluation" not in args.save_data_path_prefix:
        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "train_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(train_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "dev_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(dev_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "test_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(test_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "trainListFile.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(train_dial_id_list, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "valListFile.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(dev_dial_id_list, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "testListFile.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(test_dial_id_list, f)
        f.close()
    
    else: # save dialogues for human evaluation
        dial_idx = 0
        anoy_train_dials = {}
        anoy_train_dials_1_3 = {}
        anoy_train_dials_2_3 = {}
        anoy_train_dials_3_3 = {}
        for dial_id in train_dials:
            orig_dial = train_dials[dial_id]
            anoy_dial = []
            context = []
            for orig_turn in orig_dial:
                anoy_turn = {}
                
                anoy_turn["context"] = "\n".join(context)
                for k, v in orig_turn.items():
                    if k == "dial_id":
                        anoy_turn[k] = f"dial_{dial_idx}"
                    elif k in ['turn_num', 'user', 'bspn', 'bspn_reform', 'db', 'nodelx_resp', 'aspn']:
                        anoy_turn[k] = orig_turn[k]
                anoy_turn["bs_error, if there is error in the belief state (not consistent with user utter), fill in 1, otherwise, fill in 0"] = ""
                anoy_turn["da_error, if there is error in the dialog act (not consistent with system resp), fill in 1, otherwise, fill in 0"] = ""
                anoy_turn["grammar_error, if there is grammar error, fill in 1, otherwise, fill in 0"] = ""

                anoy_turn["user_fluency, if the user is fluent and natural, fill in 1, otherwise fill in 0"] = ""
                anoy_turn["system_fluency, if the response is fluent and natural, fill in 1, otherwise fill in 0"] = ""

                context.append(f"User: {orig_turn['user']}")
                context.append(f"System: {orig_turn['nodelx_resp']}")

                anoy_dial.append(anoy_turn)
            
            if dial_idx < 33:
                anoy_train_dials_1_3[f"dial_{dial_idx}"] = anoy_dial
            elif dial_idx < 66:
                anoy_train_dials_2_3[f"dial_{dial_idx}"] = anoy_dial
            else:
                anoy_train_dials_3_3[f"dial_{dial_idx}"] = anoy_dial

            anoy_train_dials[f"dial_{dial_idx}"] = anoy_dial
            dial_idx += 1 
        
        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "train_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(train_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "anonymous_train_raw_dials_1_3.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(anoy_train_dials_1_3, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "anonymous_train_raw_dials_2_3.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(anoy_train_dials_2_3, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "anonymous_train_raw_dials_3_3.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(anoy_train_dials_3_3, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "dev_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(dev_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "test_raw_dials.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(test_dials, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "trainListFile.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(train_dial_id_list, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "valListFile.json")
        f = open(save_dialog_turn_info_path, "w")
        json.dump(dev_dial_id_list, f)
        f.close()

        save_dialog_turn_info_path = os.path.join(args.save_data_path_prefix, "testListFile.json")

        f = open(save_dialog_turn_info_path, "w")
        json.dump(test_dial_id_list, f)
        f.close()






