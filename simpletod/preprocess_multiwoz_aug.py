# -*- coding: utf-8 -*-
import copy
import json
import os
import sys
import re
import shutil
import urllib
from urllib import request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np

from utils.multiwoz import dbPointer
from utils.multiwoz import delexicalize

from utils.multiwoz.nlp import normalize, normalize_lexical, normalize_beliefstate, normalize_mine
import ipdb

np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 1000000
MAX_LENGTH = 600

DATA_DIR = './resources_e2e_2.3_0.01_augx1/'


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def createDict(word_freqs):
    words = [k for k in word_freqs.keys()]
    freqs = [v for v in word_freqs.values()]

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    SEP0 = '_SEP0'
    SEP1 = '_SEP1'
    SEP2 = '_SEP2'
    SEP3 = '_SEP3'
    SEP4 = '_SEP4'
    SEP5 = '_SEP5'
    SEP6 = '_SEP6'
    SEP7 = '_SEP7'
    extra_tokens = [_GO, EOS, UNK, PAD, SEP0, SEP1, SEP2, SEP3, SEP4, SEP5, SEP6, SEP7]
    # extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii #+ len(extra_tokens)

    new_worddict = worddict.copy()
    for key, idx in worddict.items():
        if idx >= DICT_SIZE:
            del new_worddict[key]
    return new_worddict


def get_belief_state(bstate):
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    raw_bstate = []
    for domain in domains:
        for slot, value in bstate[domain]['semi'].items():
            if value:
                raw_bstate.append((domain, slot, normalize_beliefstate(value)))
        for slot, value in bstate[domain]['book'].items():
            if slot == 'booked':
                continue
            if value:
                new_slot = '{} {}'.format('book', slot)
                raw_bstate.append((domain, new_slot, normalize_beliefstate(value)))
    # ipdb.set_trace()
    return raw_bstate


def normalize_pptod_bs(bstate):
    # convert the bspn_reform to the bstate format in simpletod
    # example: [hotel] pricerange is cheap , type is hotel -> 
    return


def normalize_sys_act(sys_act):

    normalized_sys_act = []

    sys_act_dict = paser_aspn_to_dict(sys_act)
    for domain, domain_da in sys_act_dict.items():

        normalized_domain = domain.replace("[", "").replace("]", "").strip()

        for act, act_slots in domain_da.items():
            act = act.replace("[", "").replace("]", "").strip()
            if not act_slots:
                normalized_sys_act.append([normalized_domain, act, "none"])
            else:
                for act_slot in act_slots:
                    normalized_sys_act.append([normalized_domain, act, act_slot])

    return normalized_sys_act


def normalize_bstate(bstate):
    normalized_bstate = []

    bstate_dict = paser_bs_reform_to_dict(bstate)
    for domain, domain_bs in bstate_dict.items():

        normalized_domain = domain.replace("[", "").replace("]", "").strip()

        for slot_name, slot_value in domain_bs.items(): # split domain-slots to domains and slots

            if slot_name == 'stay':
                slot_name = 'book stay'
            if slot_name == 'day' and normalized_domain in ['hotel', 'restaurant']:
                slot_name = 'book day'
            if slot_name == 'people':
                slot_name = 'book people'
            if slot_name == 'time':
                slot_name = 'book time'
            if slot_name == 'arrive':
                slot_name = 'arriveBy'
            if slot_name == 'leave':
                slot_name = 'leaveAt'
            
            normalized_bstate.append([normalized_domain, slot_name, slot_value])

    return normalized_bstate


def paser_bs_reform_to_dict(sent):
    """Convert compacted bs span to triple list
        Ex:  
    """
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
            domain_bs = {}
        sub_span = " ".join(sent[d_idx+1:next_d_idx])
        for bs in sub_span.split(","):
            if bs and len(bs.split(" is ")) == 2:
                slot_name, slot_value = bs.split(" is ")
                slot_name = slot_name.strip()
                slot_value = slot_value.strip()
                if slot_name and slot_value:
                    domain_bs[slot_name] = slot_value
        belief_state[domain] = domain_bs
    return belief_state


def paser_aspn_to_dict(sent):

    all_domain = ["[taxi]","[police]","[hospital]","[hotel]","[attraction]","[train]","[restaurant]","[general]"]
    all_acts = ["[request]", "[inform]", "[offerbook]", "[nobook]", "[reqmore]", "[bye]", "[welcome]", "[offerbooked]", "[recommend]", "[select]", "[thank]"]
    sent = sent.split()
    dialog_act = {}
    domain_idx = [idx for idx,token in enumerate(sent) if token in all_domain+["[general]"]]
    for i,d_idx in enumerate(domain_idx):
        next_d_idx = len(sent) if i+1 == len(domain_idx) else domain_idx[i+1]
        domain = sent[d_idx]
        if domain in dialog_act:
            domain_da = dialog_act[domain]
        else:
            domain_da = {}
        sub_span = sent[d_idx+1:next_d_idx]
        sub_a_idx = [idx for idx,token in enumerate(sub_span) if token in all_acts]
        for j,a_idx in enumerate(sub_a_idx):
            next_a_idx = len(sub_span) if j+1 == len(sub_a_idx) else sub_a_idx[j+1]
            act = sub_span[a_idx]
            act_slots = sub_span[a_idx+1:next_a_idx]
            domain_da[act] = act_slots
        dialog_act[domain] = domain_da
    return dialog_act


def loadData(lexicalize=False):
    """Given test and validation sets, divide
    the data for three different sets"""
    # ipdb.set_trace()

    test_raw_dials = {}
    fin = open(os.path.join(DATA_DIR, 'multi-woz', 'test_raw_dials.json'), 'r')
    test_raw_dials = json.load(fin)
    fin.close()

    val_raw_dials = {}
    fin = open(os.path.join(DATA_DIR, 'multi-woz', 'dev_raw_dials.json'), 'r')
    val_raw_dials = json.load(fin)
    fin.close()

    train_raw_dials = {}
    fin = open(os.path.join(DATA_DIR, 'multi-woz', 'train_raw_dials.json'), 'r')
    train_raw_dials = json.load(fin)
    fin.close()

    all_raw_dials = [train_raw_dials, val_raw_dials, test_raw_dials]

    train_dials, val_dials, test_dials = {}, {}, {}
    all_dials = [train_dials, val_dials, test_dials]

    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()
    word_freqs_history = OrderedDict()

    for idx, raw_dials in enumerate(all_raw_dials):

        dials = all_dials[idx]

        for dialogue_name in tqdm(raw_dials):

            dial = raw_dials[dialogue_name]
            if dial:
                dialogue = {}
                dialogue['usr'] = []
                dialogue['sys'] = []
                dialogue['db'] = []
                dialogue['bs'] = []
                dialogue['bstate'] = []
                dialogue['sys_act_raw'] = []
                dialogue['sys_act'] = []
                for turn_id, turn in enumerate(dial):
                    
                    usr = turn['user']
                    # record word freq in user utterance and history
                    words_in = usr.strip().split(' ')
                    for w in words_in:
                        if w not in word_freqs_usr:
                            word_freqs_usr[w] = 0
                        word_freqs_usr[w] += 1
                        if w not in word_freqs_history:
                            word_freqs_history[w] = 0
                        word_freqs_history[w] += 1

                    if lexicalize:
                        sys = turn['nodelx_resp']
                    else:
                        sys = turn['resp']
                    # record word freq in system response and history
                    words_in = sys.strip().split(' ')
                    for w in words_in:
                        if w not in word_freqs_sys:
                            word_freqs_sys[w] = 0
                        word_freqs_sys[w] += 1
                        if w not in word_freqs_history:
                            word_freqs_history[w] = 0
                        word_freqs_history[w] += 1

                    if 'bspn_reform' in turn:
                        bstate = turn['bspn_reform']
                        bstate = normalize_bstate(bstate)
                    else:
                        bstate = []

                    if 'aspn_reform' in turn:
                        sys_act = turn['aspn_reform']
                        sys_act = normalize_sys_act(sys_act)
                    else:
                        sys_act = []

                    dialogue['usr'].append(usr)
                    dialogue['sys'].append(sys)
                    dialogue['bstate'].append(bstate)
                    dialogue['sys_act'].append(sys_act)

                    # deprecated in our experiments
                    dialogue['db'].append([])
                    dialogue['bs'].append([])
                    dialogue['sys_act_raw'].append([])
                
                # record the processed dialogues
                dials[dialogue_name] = dialogue

    # save all dialogues
    if lexicalize:
        val_filename = os.path.join(DATA_DIR, 'val_dials_lexicalized.json')
        test_filename = os.path.join(DATA_DIR, 'test_dials_lexicalized.json')
        train_filename = os.path.join(DATA_DIR, 'train_dials_lexicalized.json')
    else:
        val_filename = os.path.join(DATA_DIR, 'val_dials.json')
        test_filename = os.path.join(DATA_DIR, 'test_dials.json')
        train_filename = os.path.join(DATA_DIR, 'train_dials.json')
    
    with open(val_filename, 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(test_filename, 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(train_filename, 'w') as f:
        json.dump(train_dials, f, indent=4)

    return word_freqs_usr, word_freqs_sys, word_freqs_history


def buildDictionaries(word_freqs_usr, word_freqs_sys, word_freqs_histoy, lexicalize=False):
    """Build dictionaries for both user and system sides.
    You can specify the size of the dictionary through DICT_SIZE variable."""
    dicts = []
    worddict_usr = createDict(word_freqs_usr)
    dicts.append(worddict_usr)
    worddict_sys = createDict(word_freqs_sys)
    dicts.append(worddict_sys)
    worddict_history = createDict(word_freqs_histoy)
    dicts.append(worddict_history)

    # reverse dictionaries
    idx2words = []
    for dictionary in dicts:
        dic = {}
        for k, v in dictionary.items():
            dic[v] = k
        idx2words.append(dic)

    if lexicalize:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word_lexicalized.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index_lexicalized.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word_lexicalized.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index_lexicalized.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word_lexicalized.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index_lexicalized.json')
    else:
        input_index2word_filename = os.path.join(DATA_DIR, 'input_lang.index2word.json')
        input_word2index_filename = os.path.join(DATA_DIR, 'input_lang.word2index.json')
        output_index2word_filename = os.path.join(DATA_DIR, 'output_lang.index2word.json')
        output_word2index_filename = os.path.join(DATA_DIR, 'output_lang.word2index.json')
        history_index2word_filename = os.path.join(DATA_DIR, 'history_lang.index2word.json')
        history_word2index_filename = os.path.join(DATA_DIR, 'history_lang.word2index.json')

    with open(input_index2word_filename, 'w') as f:
        json.dump(idx2words[0], f, indent=2)
    with open(input_word2index_filename, 'w') as f:
        json.dump(dicts[0], f, indent=2)
    with open(output_index2word_filename, 'w') as f:
        json.dump(idx2words[1], f, indent=2)
    with open(output_word2index_filename, 'w') as f:
        json.dump(dicts[1], f, indent=2)
    with open(history_index2word_filename, 'w') as f:
        json.dump(idx2words[2], f, indent=2)
    with open(history_word2index_filename, 'w') as f:
        json.dump(dicts[2], f, indent=2)


def main():

    lexicalize = str(sys.argv[1])=='lexical'

    word_freqs_usr, word_freqs_sys, word_freqs_history = loadData(lexicalize)
    print('Building dictionaries')
    buildDictionaries(word_freqs_usr, word_freqs_sys, word_freqs_history,
                      lexicalize=(str(sys.argv[1])=='lexical'))


if __name__ == "__main__":
    main()
