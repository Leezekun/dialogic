from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
import en_core_web_sm
from nltk import ngrams
from utils.multiwoz import dbPointer
import ipdb
import json
import random
import os

from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

save_dir = './resources_e2e_2.3_0.01_augx1/gpt2'
os.makedirs(save_dir, exist_ok=True)

for split in ['train', 'val', 'test']:

    opt = ArgsParser().parse()
    opt.use_knowledge = True
    opt.use_action = True
    opt.context_knowledge = True
    opt.lexical = True
    data = MultiWozDataset(opt, split=split, shuffle=False)

    opt_delex = ArgsParser().parse()
    data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)

    history_raw_new = []
    belief_raw_new = []
    belief_raw_none_new = []
    action_raw_new = []
    output_raw_new = []
    output_raw_delex_new = []
    
    if split == 'test':
        test_dict = {}

    lex_dict = {}
    delex_dict = {}
    for d in data:
        lex_dict[d['name']] = d

    for d in data_delex:
        delex_dict[d['name']] = d

    # use delex system response for training and evaluation !!!
    for key in delex_dict:
        # d_lex = lex_dict[key]
        d_delex = delex_dict[key]
        inp = d_delex['input_raw']
        # out = d_lex['target_raw']
        out_delex = d_delex['target_raw']

        for i, (usr, sys_delex) in enumerate(zip(inp, out_delex)):
            if i == 0:
                history_new = '<|context|> <|user|> {} <|endofcontext|>'.format(usr)
            else:
                tmp_new = ['<|context|>']
                for k in range(i):

                    tmp_new.append('<|user|> ' + inp[k])
                    tmp_new.append('<|system|> ' + out_delex[k])

                tmp_new.append('<|user|> ' + usr + '<|endofcontext|>')
                history_new = ' '.join(tmp_new)

            # sys_delex = out_delex[i]
            history_raw_new.append(history_new)
            # output_raw_new.append('<|response|> ' + sys + ' <|endofresponse|>')

            output_raw_delex_new.append('<|response|> ' + sys_delex.strip() + ' <|endofresponse|>')

        belief = d_delex['belief_raw']
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if b[-1] in ['not mentioned']: # comment this for DST task
                    continue
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_new.append(tmp_new)

        # belief for DST task (include none)
        for bs in belief:
            tmp_bs_new = []
            for i, b in enumerate(bs):
                if i == len(bs) - 1:
                    tmp_bs_new.append(' '.join(b))
                else:
                    tmp_bs_new.append(' '.join(b))

            if len(tmp_bs_new) == 0:
                tmp_bs_new.append(' ')

            tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
            belief_raw_none_new.append(tmp_new)

        action = d_delex['action_raw']
        for act in action:
            tmp_act_new = []
            for i, a in enumerate(act):
                if i == len(act) - 1:
                    tmp_act_new.append(' '.join(a))
                else:
                    tmp_act_new.append(' '.join(a))
            if len(tmp_act_new) == 0:
                tmp_act_new.append(' ')

            tmp_new = '<|action|> {} <|endofaction|>'.format(' , '.join(tmp_act_new))
            action_raw_new.append(tmp_new)

    # this is what we do dst experiments on
    tmp = []
    for hist, bs in zip(history_raw_new, belief_raw_none_new):
        tmp.append(' '.join([hist.lower(), bs.lower()]))
    with open('{}/{}.history_belief'.format(save_dir, split),
              'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))
    
    # this is what we do e2e experiments on
    tmp = []
    for hist, bs, act, trg in zip(history_raw_new, belief_raw_new, action_raw_new, output_raw_delex_new):
        tmp.append(' '.join([hist.lower(), bs.lower(), act, trg]))
    with open('{}/{}.history_belief_action_sys_delex'.format(save_dir, split), 'wt') as f:
        for l in tmp:
            f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))

