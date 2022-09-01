
from ast import arg
import enum
from pickle import NONE
from tkinter import dialog
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
from regex import F

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
    parser.add_argument('--use_db_as_input', type=str, default='False', 
        help="True or False, whether includes db result as part of the input when generating response.")
    parser.add_argument('--cascaded', type=str, default='False', 
        help="True or False, whether includes action when generating response.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    
    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    parser.add_argument('--aug_train_data_file', type=str, default="", help='the augmented data file path')


    return parser.parse_args()


import argparse
if __name__ == '__main__':
    args = parse_config()
    
    print ('Start loading data...')
    from dataclass import MultiWozData

    if args.data_version == "2.0":
        from config import Config
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-dialogic-processed")

    elif args.data_version == "2.1":
        from config21 import Config
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.1-dialogic-processed")

    elif args.data_version == "2.3":
        from config23 import Config
        save_output_path = os.path.join(args.data_path_prefix, "multi-woz-2.3-dialogic-processed")

    elif args.data_version == "2.4":
        from config24 import Config
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
        data_mode='interact', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio, aug_train_data_file=args.aug_train_data_file)
    else:
        data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
        data_mode='interact', data_version=args.data_version, use_db_as_input=use_db_as_input, cascaded=cascaded, add_special_decoder_token=add_special_decoder_token, 
        train_data_ratio=args.train_data_ratio)
        
    ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
    input_contain_db=use_db_as_input
    eva_batch_size = -1 # -1 indicate no batch, just load all

    # print("Counting the possible slot values......")
    # for eva_mode in ['train', 'dev', 'test', 'all']:
        
    #     dev_batch_list = data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
    #         eva_batch_size=eva_batch_size, eva_mode=eva_mode)  
    #     dialogs_list = dev_batch_list[-1]

    #     # reorganize the list into dict
    #     ontology_dict = {}
    #     for turn in dialogs_list:
    #         
        
    #     # remove repeat slot values
    #     for domain in ontology_dict:
    #         for slot in ontology_dict[domain]:
    #             possible_slot_valbs_reform = turn['bspn_reform']
    #         bs_dict = paser_bs_reform_to_dict(bs_reform)
    #         for domain, domain_bs in bs_dict.items():
    #             if domain in ontology_dict:
    #                 domain_ontology = ontology_dict[domain]
    #             else:
    #                 domain_ontology = {}
    #             for slot, slot_value in domain_bs.items():
    #                 if any ([_ in slot_value for _ in ["none", "null"]]) or (re.search("\W", slot_value) and ":" not in slot_value):
    #                     continue
    #                 if slot in domain_ontology:
    #                     slot_ontology = domain_ontology[slot]
    #                 else:
    #                     slot_ontology = []
    #                 slot_ontology.append(slot_value)
    #                 domain_ontology[slot] = slot_ontology
    #             ontology_dict[domain] = domain_ontologyues = list(set(ontology_dict[domain][slot]))
    #             ontology_dict[domain][slot] = possible_slot_values

    #     print(f"Ontology for {eva_mode}: {ontology_dict}") 

    #     dialogs_analysis[eva_mode] = ontology_dict            

    # print("Start saving the possible slot values......")
    # # save the dialog turn info
    # assert save_output_path is not None
    # save_dialog_turn_info_path = os.path.join(save_output_path, "possible_slot_values.json")
    # f = open(save_dialog_turn_info_path, "w")
    # json.dump(dialogs_analysis, f)
    # f.close()

    print("Counting stats......")
    raw_train_dialogs = data.convert_to_raw_train_data()

    unique_user_tokens = []
    unique_resp_tokens = []
    unique_user_3grams = []
    unique_resp_3grams = []
    unique_bs = []
    unique_da = []

    total_turn_num = 0
    total_dialog_num = len(raw_train_dialogs)
    total_domain = 0
    total_informable_slot = 0
    total_requestable_slot = 0
    total_dialog_booked = 0

    if not args.aug_train_data_file:
        half = True
    else:
        half = False

    for dial_idx, dial_id in enumerate(raw_train_dialogs):
        if half and dial_idx < total_dialog_num / 2:
            continue

        raw_dialog = raw_train_dialogs[dial_id]
        dialog_domain = []
        dialog_informable_slot = []
        dialog_requestable_slot = []

        for turn in raw_dialog:
            user = turn['user'].split()
            resp = turn['resp'].split()
            domains = turn['turn_domain']
            bs_reform = turn['bspn_reform']
            aspn = turn['aspn']

            unique_user_tokens.extend(user)
            unique_resp_tokens.extend(resp)
            
            if len(user) >= 3:
                for i in range(len(user)-2):
                    unique_user_3grams.append(" ".join(user[i:i+3]))
            if len(resp) >= 3:
                for i in range(len(resp)-2):
                    unique_resp_3grams.append(" ".join(resp[i:i+3]))

            for domain in domains:
                if domain not in ['[general]']:
                    dialog_domain.append(domain)
            
            bs_dict = paser_bs_reform_to_dict(bs_reform)
            for domain, domain_bs in bs_dict.items():
                if domain not in ['[general]']:
                    for slot, slot_value in domain_bs.items():
                        dialog_informable_slot.append(f"{domain}-{slot}-{slot_value}")
            
            da_dict = paser_aspn_to_dict(aspn)
            for domain, domain_da in da_dict.items():
                for act, slot in domain_da.items():
                    if "inform" in act:
                        dialog_requestable_slot.append(f"{domain}-{act}-{slot}")
                    if "offerbooked" in act:
                        total_dialog_booked += 1
                        
            total_turn_num += 1

        dialog_domain = list(set(dialog_domain))
        dialog_informable_slot = list(set(dialog_informable_slot))
        # dialog_requestable_slot = list(set(dialog_requestable_slot))

        unique_bs.extend(dialog_informable_slot)
        unique_da.extend(dialog_requestable_slot)

        total_domain += len(dialog_domain)
        total_informable_slot += len(dialog_informable_slot)
        total_requestable_slot += len(dialog_requestable_slot)

    if half:
        total_dialog_num = int(total_dialog_num / 2)

    total_user_len = len(unique_user_tokens)
    total_resp_len = len(unique_resp_tokens)

    avg_user_len = total_user_len / total_turn_num
    avg_resp_len = total_resp_len / total_turn_num

    avg_domain = total_domain / total_dialog_num
    avg_informable_slot = total_informable_slot / total_dialog_num
    avg_requestable_slot = total_requestable_slot / total_dialog_num

    unique_user_tokens = set(unique_user_tokens)
    unique_resp_tokens = set(unique_resp_tokens)
    unique_user_3grams = set(unique_user_3grams)
    unique_resp_3grams = set(unique_resp_3grams)
    unique_bs = set(unique_bs)
    unique_da = set(unique_da)

    user_tokens_num = len(unique_user_tokens)
    resp_tokens_num = len(unique_resp_tokens)
    user_3grams_num = len(unique_user_3grams)
    resp_3grams_num = len(unique_resp_3grams)

    print(f"Total dialogs, {total_dialog_num}, turns: {total_turn_num}, domain: {total_domain}, booked domain: {total_dialog_booked}, informed slot: {total_informable_slot}, requestable slot: {total_requestable_slot}.")     
    print(f"Avg user length: {avg_user_len}, resp length: {avg_resp_len}, domain: {avg_domain}, informed slot: {avg_informable_slot}, requestable slot: {avg_requestable_slot} ")
    print(f"Total response tokens: {resp_tokens_num}, 3-grams: {resp_3grams_num}")
    print(len(unique_bs), len(unique_da))

