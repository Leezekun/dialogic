import re
def restore_text(text):
    text = re.sub('=', '', text)
    text = re.sub(' is ', ' ', text)
    text = re.sub(',', '', text)
    text = ' '.join(text.split()).strip()
    return text

def split_bs_text_by_domain(bs_text, bsdx_text):
    res_text_list = []
    bs_text = bs_text.strip('<sos_b>').strip('<eos_b>').strip()
    bsdx_text = bsdx_text.strip('<sos_b>').strip('<eos_b>').strip()
    token_list = bsdx_text.split()
    domain_list = []
    for token in token_list:
        if token.startswith('[') and token.endswith(']'):
            domain_list.append(token)
    if domain_list == 1: # only have one domain
        return [bs_text], [bsdx_text]
    else:
        bs_list, bsdx_list = [], []
        for idx in range(len(domain_list)):
            curr_domain = domain_list[idx]
            if idx == len(domain_list)-1: # last domain
                bs_text_snippet = curr_domain + ' ' + bs_text.split(curr_domain)[1].strip()
                bsdx_text_snippet = curr_domain + ' ' + bsdx_text.split(curr_domain)[1].strip()
            else:
                next_domain = domain_list[idx+1]
                bs_text_snippet = curr_domain + ' ' + bs_text.split(curr_domain)[1].split(next_domain)[0]
                bsdx_text_snippet = curr_domain + ' ' + bsdx_text.split(curr_domain)[1].split(next_domain)[0]
            bs_list.append(bs_text_snippet.strip())
            bsdx_list.append(bsdx_text_snippet.strip())
    return bs_list, bsdx_list

def parse_bs_bsdx(bs_text, bsdx_text):
    # this function deals belief state from single domain
    # we assume there is no repitive slots in the bsdx text
    dx_token_list = bsdx_text.split()

    bs_name_list = bsdx_text.split()
    token_num = len(bs_name_list)
    map_dict = {}
    res_bs_text = ''
    res_bsdx_text = ''
    for idx in range(token_num):
        curr_slot = bs_name_list[idx]
        if curr_slot.startswith('[') and curr_slot.endswith(']'):
            continue
        else:
            if idx == token_num - 1:
                #curr_value = bs_text.split(' ' + curr_slot + ' ')[-1].strip()
                curr_value = bs_text.split(' ' + curr_slot + ' ')[-1].strip()
            else:
                next_slot = bs_name_list[idx+1]
                curr_value = bs_text.split(' ' + curr_slot + ' ')[1].split(' ' + next_slot)[0].strip()
            map_dict[curr_slot] = curr_value
    for curr_slot in bs_name_list:
        if curr_slot.startswith('[') and curr_slot.endswith(']'):
            res_bs_text += curr_slot + ' '
            res_bsdx_text += curr_slot + ' '
        else:
            res_bs_text += curr_slot + ' is ' + map_dict[curr_slot] + ' , '
            res_bsdx_text += curr_slot + ' , '
    
    res_bs_text = res_bs_text.strip().strip(',').strip()
    res_bsdx_text = res_bsdx_text.strip().strip(',').strip()
    return ' '.join(res_bs_text.split()).strip(), ' '.join(res_bsdx_text.split()).strip()

def overall_parsing_bs_bsdx(bs_text, bsdx_text):
    in_bs_text, in_bsdx_text = bs_text, bsdx_text
    if bs_text == '<sos_b> <eos_b>':
        assert bsdx_text == '<sos_b> <eos_b>'
        return bs_text, bsdx_text
    bs_text_list, bsdx_text_list = split_bs_text_by_domain(bs_text, bsdx_text)
    res_bs_text, res_bsdx_text = ' ', ' '
    for idx in range(len(bs_text_list)):
        one_bs_text, one_bsdx_text = parse_bs_bsdx(bs_text_list[idx], bsdx_text_list[idx])
        res_bs_text += one_bs_text + ' '
        res_bsdx_text += one_bsdx_text + ' '
    res_bs_text = '<sos_b> ' + res_bs_text.strip() + ' <eos_b>'
    res_bsdx_text = '<sos_b> ' + res_bsdx_text.strip() + ' <eos_b>'
    assert restore_text(res_bs_text) == in_bs_text
    assert restore_text(res_bsdx_text) == in_bsdx_text
    return res_bs_text,  res_bsdx_text

def parse_action_text(text):
    in_text = text
    res_text = ''
    token_list = text.strip('<sos_a>').strip('<eos_a>').strip().split()
    token_num = len(token_list)
    for idx in range(token_num):
        curr_token = token_list[idx]
        if curr_token.startswith('[') and curr_token.endswith(']'):
            res_text += curr_token + ' '
        else:
            if idx == token_num - 1:
                res_text += curr_token
            else:
                next_token = token_list[idx + 1]
                if next_token.startswith('[') and next_token.endswith(']'):
                    res_text += curr_token + ' '
                else:
                    #res_text += curr_token + ' , '
                    res_text += curr_token + ' '
    res_text = '<sos_a> ' + ' '.join(res_text.split()).strip() + ' <eos_a>'
    assert restore_text(res_text) == in_text
    return res_text

def tokenized_decode(token_id_list, tokenizer, special_token_list):
    pred_tokens = tokenizer.convert_ids_to_tokens(token_id_list)
    res_text = ''
    curr_list = []
    for token in pred_tokens:
        if token in special_token_list:
            if len(curr_list) == 0:
                res_text += ' ' + token + ' '
            else:
                curr_res = tokenizer.convert_tokens_to_string(curr_list)
                res_text = res_text + ' ' + curr_res + ' ' + token + ' '
                curr_list = []
        else:
            curr_list.append(token)
    if len(curr_list) > 0:
        curr_res = tokenizer.convert_tokens_to_string(curr_list)
        res_text = res_text + ' ' + curr_res + ' '
    res_text_list = res_text.strip().split()
    res_text = ' '.join(res_text_list).strip()
    return res_text

def transform_id_dict(in_dict, tokenizer, special_token_list):
    res_dict = {}
    for key in in_dict:
        if key in ['pointer','turn_num','turn_domain','dial_id']:
            res_dict[key] = in_dict[key]
        else:
            one_text = tokenized_decode(in_dict[key], tokenizer, special_token_list)
            one_text = ' '.join(one_text.split()).strip()
            res_dict[key] = one_text
    return res_dict

def process_dict(in_dict, tokenizer, special_token_list):
    in_dict = transform_id_dict(in_dict, tokenizer, special_token_list)
    res_dict = in_dict.copy()
    bs_text_reform, bsdx_text_reform = overall_parsing_bs_bsdx(in_dict['bspn'], in_dict['bsdx'])
    res_dict['bspn_reform'] = ' '.join(bs_text_reform.split()).strip()
    res_dict['bsdx_reform'] = ' '.join(bsdx_text_reform.split()).strip()
    action_text_reform = parse_action_text(in_dict['aspn'])
    res_dict['aspn_reform'] = ' '.join(action_text_reform.split()).strip()
    return res_dict

import progressbar
def process_data_list(data_list, tokenizer, special_token_list):
    res_list = []
    p = progressbar.ProgressBar(len(data_list))
    p.start()
    p_idx = 0
    for item in data_list:
        p_idx += 1
        p.update(p_idx)
        try:
            one_res_list = [process_dict(sub_item, tokenizer, special_token_list) for sub_item in item]
        except: 
            continue
        res_list.append(one_res_list)
    p.finish()
    print (len(res_list), len(data_list))
    return res_list

if __name__ == '__main__':
    import json
    import os
    os.mkdir(r'../data/multi-woz-2.3-fine-processed/')

    from config23 import Config
    data_prefix = r'../data/'
    cfg = Config(data_prefix)

    model_path = 't5-small'
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    import ontology
    in_f = r'../special_token_list.txt'
    special_token_list = ontology.sos_eos_tokens
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            token = l.strip('\n')
            if token.startswith('[value'):
                special_token_list.append(token)
            else:
                pass
    tokenizer.add_tokens(special_token_list)
    print (special_token_list)

    print ('Starting loading raw data...')
    from reader import MultiWozReader
    reader = MultiWozReader(tokenizer, cfg, data_mode='train')
    print ('Raw data loaded.')

    print ('Start processing training data...')
    train_list = process_data_list(reader.train, tokenizer, special_token_list)
    out_f = r'../data/multi-woz-2.3-fine-processed/multiwoz-fine-processed-train.json'
    with open(out_f, 'w') as outfile:
        json.dump(train_list, outfile, indent=4)
    print ('Training data saved!')

    print ('Start processing dev data...')
    dev_list = process_data_list(reader.dev, tokenizer, special_token_list)
    out_f = r'../data/multi-woz-2.3-fine-processed/multiwoz-fine-processed-dev.json'
    with open(out_f, 'w') as outfile:
        json.dump(dev_list, outfile, indent=4)
    print ('Dev data saved!')

    print ('Start processing test data...')
    test_list = process_data_list(reader.test, tokenizer, special_token_list)
    out_f = r'../data/multi-woz-2.3-fine-processed/multiwoz-fine-processed-test.json'
    with open(out_f, 'w') as outfile:
        json.dump(test_list, outfile, indent=4)
    print ('Test data saved!')












