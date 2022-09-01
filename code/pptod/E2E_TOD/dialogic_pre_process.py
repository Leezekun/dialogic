
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

from dialogic_utils import system_prefix, user_prefix
from dialogic_utils import *


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration    
    parser.add_argument('--multiwoz_schema_path', type=str, default="../data/multiwoz/data/multi-woz-2.3-fine-processed/schema.json", help='the path that stores the schema for multiwoz dataset.')
    parser.add_argument('--dialog_message_path', type=str, default="../data/multiwoz/data/multi-woz/data.json", help='the path that stores the error test cases.')
    
    parser.add_argument('--data_path_prefix', type=str, default='../data/multiwoz/data', help='The path where the data stores.')
    parser.add_argument('--data_version', type=str, default='2.3', help='The version of used multiwoz data, 2.0, 2.1, 2.3, 2.4')
    
    parser.add_argument('--model_name', type=str, default='t5-small', 
        help="the model type of t5, t5-small, t5-base, or t5-large.")

    parser.add_argument('--shuffle_mode', type=str, default='unshuffle', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--use_db_as_input', type=str, default='False', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--cascaded', type=str, default='False', 
        help="True or False, whether includes action when generating response.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    
    parser.add_argument('--eva_mode', type=str, default='train', 
        help="test or dev, or train, or all, evaluation on test or dev dataset")
    parser.add_argument('--train_data_ratio', type=float, default=0.01, help='the ratio of training data used for training the model')

    return parser.parse_args()


def parse_user_act(sent):
    all_domain = ["[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]","[general]"]
    sent = sent.split()
    belief_state = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain] 
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in belief_state:
            domain_bs = belief_state[domain]
        else:
            domain_bs = []
        sub_span = sent[d_idx+1:next_d_idx]
        for bs in sub_span:
            if bs not in domain_bs:
                domain_bs.append(bs)
        belief_state[domain] = domain_bs
    return belief_state


def parse_dialog_turns(dialog_turns, dialogs_with_user_act):

    # parse this dialog
    dialog_bs = {}
    dialog_domains = []
    turn_info_dict = {}
    turn_info_list = []

    user_act_list = dialogs_with_user_act['log']

    """
    parse this turn:
        1. find the new bs in this turn
        2. update the tracked bs
    """
    error_turn_num = 0
    orig_error_turn_num = 0
    error_correct_turn_num = 0

    turn_num = len(list(dialog_turns.keys()))
    for turn_id in range(turn_num):
        # read the information from the demo dialog
        turn = dialog_turns[turn_id]
        user = turn['user']
        response = turn['resp']
        nodelx_response = turn['nodelx_resp']
        action = turn['aspn']

        error_turn = False
        orig_error_turn = False
        error_correct_turn = False

        # parse each turn's bs based on bspn_reform
        bspn_reform = turn['bspn_reform']
        gt_goal = paser_bs_reform_to_dict(bspn_reform)

        # record turn_bs and turn_domain
        turn_bs = {}
        
        turn_domain = turn['turn_domain']
        # add this new domain, includes [general]
        for domain in turn_domain:
            
            if domain == "[general]": # normally indicate the end of a dialog
                dialog_bs[domain] = {}
                turn_bs[domain] = {}

            else:
                # obtain the current bs for the domain/turn (dict)
                if domain in dialog_bs:
                    dialog_domain_bs = copy.deepcopy(dialog_bs[domain])
                else:
                    dialog_domain_bs = {}
                if domain in turn_bs:
                    turn_domain_bs = copy.deepcopy(turn_bs[domain])
                else:
                    turn_domain_bs = {}

                user_act = user_act_list[turn_id]['user_act']
                if user_act: # [hotel] type stars 
                    user_act_dict = parse_user_act(user_act) # {'[hotel]': [type, stars]}
                    if domain in user_act_dict:
                        mentioned_slots = user_act_dict[domain]

                        if domain in gt_goal:
                            gt_domain_bs = gt_goal[domain]
                            # update mentioned slots
                            for slot_name in mentioned_slots:
                                if slot_name in gt_domain_bs:
                                    slot_value = gt_domain_bs[slot_name]
                                    # update
                                    dialog_domain_bs[slot_name] = slot_value
                                    turn_domain_bs[slot_name] = slot_value

                # finish this domain, update dialog_bs and turn_bs
                dialog_bs[domain] = dialog_domain_bs
                turn_bs[domain] = turn_domain_bs

        turn_bspn = paser_dict_to_bs(turn_bs)
        turn_bspn_reform = paser_dict_to_bs_reform(turn_bs)

        # detect if errors exist in this turn
        error_turn = detect_error_turn(user, turn_bs)

        if error_turn: error_turn_num += 1
        if orig_error_turn: orig_error_turn_num += 1
        if error_correct_turn: error_correct_turn_num += 1

        turn_info = {}
        turn_info['user'] = user
        turn_info['bs'] = turn_bs
        turn_info['bs_error']  = error_turn
        turn_info['bspn'] = turn_bspn
        turn_info['bspn_reform'] = turn_bspn_reform
        turn_info['aspn'] = action
        turn_info['resp'] = response
        turn_info['nodelx_resp'] = nodelx_response
        turn_info['turn_domain'] = turn_domain
        turn_info_dict[turn_id] = turn_info
        turn_info_list.append(turn_info)

    return turn_info_dict, turn_info_list, dialog_bs, turn_num, error_turn_num, orig_error_turn_num, error_correct_turn_num


def construct_dialog_with_single_turn_info(dialogs_with_messages, dialogs_with_user_acts):
    dialogs_with_turn_info = {}
    single_turn_info = []

    total_turn_num = 0
    total_error_turn_num = 0
    total_orig_error_turn_num = 0
    total_error_correct_turn_num = 0

    for dial_id, dialog_with_message in dialogs_with_messages.items():

        if f'{dial_id}.json' in dialogs_with_user_acts:
            dialogs_with_user_act = dialogs_with_user_acts[f'{dial_id}.json']
        elif dial_id in dialogs_with_user_acts:
            dialogs_with_user_act = dialogs_with_user_acts[dial_id]
        else:
            print(f"{dial_id} file not exist!")
            continue

        dialog_message = dialog_with_message['message']
        dialog_turns = dialog_with_message['dialog']

        turn_info_dict, turn_info_list, dialog_bs, turn_num, error_turn_num, \
        orig_error_turn_num, error_correct_turn_num = parse_dialog_turns(dialog_turns, dialogs_with_user_act)

        dialogs_with_turn_info[dial_id] = {"message": dialog_message, "goal": dialog_bs, "orig_turns": dialog_turns, "info_turns": turn_info_dict}
        single_turn_info.extend(turn_info_list)

        total_turn_num += turn_num
        total_error_turn_num += error_turn_num
        total_orig_error_turn_num += orig_error_turn_num
        total_error_correct_turn_num += error_correct_turn_num

        error_correct_rate = total_error_correct_turn_num / total_orig_error_turn_num if total_orig_error_turn_num > 0 else 0
        correct_rate = (total_turn_num - total_error_turn_num) / total_turn_num if total_turn_num > 0 else 0

    print(f"Total {total_turn_num} turns, {total_error_turn_num} error turns, {total_orig_error_turn_num} original error turns, {total_error_correct_turn_num} error corrected turns. Error correct rate: {error_correct_rate}. Correct rate: {correct_rate}.")

    return dialogs_with_turn_info, single_turn_info


def format_multiwoz_message(message):
    """
    should be consistent with construct_augment_dialog function
    """
    message = ". ".join(message)
    message = message.replace("<span class='emphasis'>", "*")
    message = message.replace("</span>", "*")
    return message


import argparse
if __name__ == '__main__':
    args = parse_config()
    
    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        from config import Config
        dialog_user_acts_path = os.path.join(args.data_path_prefix, "multi-woz-processed", "data_for_damd.json")
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-dialogic-processed")

    elif args.data_version == "2.1":
        from config21 import Config
        dialog_user_acts_path = os.path.join(args.data_path_prefix, "multi-woz-2.1-processed", "data_for_damd.json")
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.1-dialogic-processed")

    elif args.data_version == "2.3":
        from config23 import Config
        dialog_user_acts_path = os.path.join(args.data_path_prefix, "multi-woz-2.3-processed", "data_for_damd.json")
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.3-dialogic-processed")

    elif args.data_version == "2.4":
        from config24 import Config
        dialog_user_acts_path = os.path.join(args.data_path_prefix, "multi-woz-2.4-processed", "data_for_damd.json")
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.4-dialogic-processed")

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

    if args.eva_mode == 'dev':
        eva_mode = 'dev'
    elif args.eva_mode == 'test':
        eva_mode = 'test'
    elif args.eva_mode == 'train':
        eva_mode = 'train'
    elif args.eva_mode == 'all':
        eva_mode = 'all'
    else:
        raise Exception('Wrong Evaluation Mode!!!')

    if args.train_data_ratio > 1:
        raise Exception('Wrong Evaluation Mode!!!')
    elif args.train_data_ratio < 0:
        raise Exception('Wrong Evaluation Mode!!!')
    else:
        train_data_ratio = args.train_data_ratio

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

    if eva_mode == 'train':
        one_dev_str = f"{eva_mode}_ratio_{train_data_ratio}"
    else:
        one_dev_str = f"{eva_mode}"
    
    # load dialog messages
    dialog_messages = {}
    print("Start loading dialog messages......")
    assert args.dialog_message_path is not None
    f = open(args.dialog_message_path, "r")
    message_data = json.load(f)
    f.close()

    for dial_id, dialog in message_data.items():
        dial_id = dial_id.lower().split(".json")[0]
        if 'message' in dialog['goal']:
            # format the message
            message = dialog['goal']['message']
            message = format_multiwoz_message(message)
            dialog_messages[dial_id] = message
    print(f"Total {len(dialog_messages)} dialog messages!")

    # load single turn use acts
    assert dialog_user_acts_path is not None
    f = open(dialog_user_acts_path, "r")
    dialogs_with_user_acts = json.load(f)
    f.close()

    # load dialogs
    data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='train', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=train_data_ratio)
    ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
    input_contain_db=use_db_as_input
    eva_batch_size = -1 # -1 indicate no batch, just load all
    dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
        eva_batch_size=eva_batch_size, eva_mode=eva_mode)  
    dialogs_list = dev_batch_list[-1]

    # reorganize the list into dict
    dialogs_dict = {}
    for turn in dialogs_list:
        dial_id = turn['dial_id']
        turn_id = turn['turn_num']
        if dial_id not in dialogs_dict:
            dialogs_dict[dial_id] = {}
        dialogs_dict[dial_id][turn_id] = turn
    print ('Dialogue data loaded!')

    # add message to dialogs
    dialogs_with_messages = {}
    for dial_id, dialog in dialogs_dict.items():
        message = dialog_messages[dial_id]
        if not message:
            continue
        dialog_with_message = {}
        dialog_with_message["dialog"] = dialog
        dialog_with_message["message"] = message
        dialogs_with_messages[dial_id] = dialog_with_message
    
    print("Start constructing single turn infos......")
    dialogs_with_turn_info, single_turn_info = construct_dialog_with_single_turn_info(dialogs_with_messages, dialogs_with_user_acts)

    # save the dialog turn info
    assert save_output_path is not None
    if not os.path.exists(save_output_path):
        os.mkdir(save_output_path)

    print("Start saving the dialogs with single turn infos......")
    save_dialog_turn_info_path = os.path.join(save_output_path, "dialog_turn_info_" + one_dev_str + ".json")
    f = open(save_dialog_turn_info_path, "w")
    json.dump(dialogs_with_turn_info, f)
    f.close()
    
    print("Start saving single turn infos......")
    save_single_turn_info_path = os.path.join(save_output_path, "single_turn_info_" + one_dev_str + ".json")
    f = open(save_single_turn_info_path, "w")
    json.dump(single_turn_info, f)
    f.close()
    
    