from fileinput import filename
import json
from sklearn.metrics import f1_score, accuracy_score
import sys
import numpy as np
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs, remove_not_mentioned_name
import argparse

def compute_jacc(data,default_cleaning_flag=True,type2_cleaning_flag=False,type3_cleaning_flag=False,type4_cleaning_flag=False):
    num_turns = 0
    joint_acc = 0

    over_gen_rate = 0
    de_gen_rate = 0

    clean_tokens = ['<|endoftext|>', ]
    error_dials = {}
    for file_name in data:

        # record if this dialog is totally correct
        dial_num_turns = 0
        dial_joint_acc = 0

        dial_over_gen_rate = 0
        dial_de_gen_rate = 0

        # previous context
        previous_context = []

        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)
            turn_user = turn_data['user']
            previous_context.append(turn_user)

            # for type 3 and 4 cleaning
            turn_user = turn_data['user']
            if turn_id+1 < len(data[file_name]):
                next_turn_data = data[file_name][turn_id+1]
                next_turn_target = next_turn_data['bspn']
                next_turn_target = paser_bs(next_turn_target)
            else:
                next_turn_target = []

            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            # turn_target = remove_not_mentioned_name(previous_context, turn_target)
            turn_pred, turn_target = ignore_none(turn_pred, turn_target)

            # MultiWOZ default cleaning
            if default_cleaning_flag:
                turn_pred, turn_target = default_cleaning(turn_pred, turn_target)

            join_flag = False
            if set(turn_target) == set(turn_pred):
                joint_acc += 1
                dial_joint_acc += 1
                join_flag = True
            
            elif type2_cleaning_flag: # check for possible Type 2 noisy annotations
                flag = True
                for bs in turn_target:
                    if bs not in turn_pred:
                        de_gen_rate += 1
                        dial_de_gen_rate += 1
                        flag = False
                        break
                for bs in turn_pred:
                    if bs not in turn_target:
                        over_gen_rate += 1
                        dial_over_gen_rate += 1
                        flag = False
                        break

                if flag: # model prediction might be correct if found in Type 2 list of noisy annotations
                    dial_name = file_name.split('.')[0].upper()
                    if dial_name in IGNORE_TURNS_TYPE2 and turn_id in IGNORE_TURNS_TYPE2[dial_name]: # ignore these turns
                        pass
                    else:
                        joint_acc += 1
                        dial_joint_acc += 1
                        join_flag = True
            
            elif type3_cleaning_flag: # check for possible Type 3 noisy annotations, the late markup
                flag = True
                for bs in turn_target: 
                    if bs not in turn_pred:
                        flag = False
                        break
                if flag:
                    for bs in turn_pred:
                        if bs not in turn_target:
                            if len(bs.split()) == 3:
                                domain, slot_name, slot_value = bs.split() # bs: domain slot_name slot_value
                                user_words = [word.strip() for word in turn_user.split()]
                                if slot_value != "yes" and ((len(slot_value)==1 and slot_value in user_words) or (len(slot_value)>1 and slot_value in turn_user)):
                                    pass
                                elif slot_value == "yes" and slot_name == 'internet' and any([_ in turn_user for _ in ['wifi', 'internet']]):
                                    pass
                                elif slot_value == "yes" and slot_name == 'parking' and 'parking' in turn_user:
                                    pass
                                else:
                                    flag = False
                                    break
                            else:
                                print(bs)
                if flag:
                    # type3_cleaning acc
                    if type4_cleaning_flag:
                        joint_acc += 1
                        dial_joint_acc += 1
                    else:
                        for bs in turn_pred:
                            if bs not in next_turn_target:
                                flag = False
                                break
                        # type3_cleaning acc
                        if flag:
                            joint_acc += 1
                            dial_joint_acc += 1

            if not join_flag:
                turn_data['gtbs'] = turn_target
                turn_data['predbs'] = turn_pred

            num_turns += 1
            dial_num_turns += 1
        
        if dial_joint_acc < dial_num_turns:
            error_dials[file_name] = data[file_name]

    joint_acc /= num_turns
    over_gen_rate /= num_turns
    de_gen_rate /= num_turns

    print('joint accuracy: {}'.format(joint_acc), 'over-generation accuracy: {}'.format(over_gen_rate), 'de-generation accuracy: {}'.format(de_gen_rate))
    return joint_acc, over_gen_rate, de_gen_rate, error_dials
    