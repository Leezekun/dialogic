import math
# import utils.delexicalize as delex
from utils.multiwoz import delexicalize as delex
from collections import Counter
from nltk.util import ngrams
import json
from utils.multiwoz.nlp import normalize, normalize_for_sql
import sqlite3
import os
import random
import logging
from utils.multiwoz.nlp import BLEUScorer
import ipdb
import sys
from tqdm import tqdm

def remove_model_mismatch_and_db_data(dial_name, target_beliefs, pred_beliefs, domain, t):
    if domain == 'hotel':
        if domain in target_beliefs[t]:
            if 'type' in pred_beliefs[t][domain] and 'type' in target_beliefs[t][domain]:
                if pred_beliefs[t][domain]['type'] != target_beliefs[t][domain]['type']:
                    pred_beliefs[t][domain]['type'] = target_beliefs[t][domain]['type']
            elif 'type' in pred_beliefs[t][domain] and 'type' not in target_beliefs[t][domain]:
                del pred_beliefs[t][domain]['type']

    if 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'pizza hut fenditton':
        pred_beliefs[t][domain]['name'] = 'pizza hut fen ditton'

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'riverside brasserie':
        pred_beliefs[t][domain]['food'] = "modern european"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'charlie chan':
        pred_beliefs[t][domain]['area'] = "centre"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'saint johns chop house':
        pred_beliefs[t][domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'pizza hut fen ditton':
        pred_beliefs[t][domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'cote':
        pred_beliefs[t][domain]['pricerange'] = "expensive"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'cambridge lodge restaurant':
        pred_beliefs[t][domain]['food'] = "european"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'cafe jello gallery':
        pred_beliefs[t][domain]['food'] = "peking restaurant"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'nandos':
        pred_beliefs[t][domain]['food'] = "portuguese"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'yippee noodle bar':
        pred_beliefs[t][domain]['pricerange'] = "moderate"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'copper kettle':
        pred_beliefs[t][domain]['food'] = "british"

    if domain == 'restaurant' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] in ['nirala', 'the nirala']:
        pred_beliefs[t][domain]['food'] = "indian"

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'vue cinema':
        if 'type' in pred_beliefs[t][domain]:
            del pred_beliefs[t][domain]['type']

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'funky fun house':
        pred_beliefs[t][domain]['area'] = 'dontcare'

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'little seoul':
        pred_beliefs[t][domain]['name'] = 'downing college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'byard art':
        pred_beliefs[t][domain]['type'] = 'museum'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'trinity college':
        pred_beliefs[t][domain]['type'] = 'college'  # correct name in turn_belief_pred

    if domain == 'attraction' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain][
        'name'] == 'cambridge university botanic gardens':
        pred_beliefs[t][domain]['area'] = 'centre'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'lovell lodge':
        pred_beliefs[t][domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'whale of a time':
        pred_beliefs[t][domain]['type'] = 'entertainment'  # correct name in turn_belief_pred

    if domain == 'hotel' and 'name' in pred_beliefs[t][domain] and pred_beliefs[t][domain]['name'] == 'a and b guest house':
        pred_beliefs[t][domain]['parking'] = 'yes'  # correct name in turn_belief_pred

    if dial_name == 'MUL0116.json' and domain == 'hotel' and 'area' in pred_beliefs[t][domain]:
        del pred_beliefs[t][domain]['area']

    return pred_beliefs


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # hypothesis = [hypothesis]
        # corpus = [corpus]
        # ipdb.set_trace()

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            if type(hyps[0]) is list:
               hyps = [hyp.split() for hyp in hyps[0]]
            else:
               hyps = [hyp.split() for hyp in hyps]
            #
            refs = [ref.split() for ref in refs]
            # hyps = [hyps]
            # hyps = hyps
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            # ipdb.set_trace()
            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class MultiWozDB(object):
    # loading databases
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__)

    for domain in domains:
        db = os.path.join('utils/multiwoz/db/{}-dbase.db'.format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    def queryResultVenues(self, domain, turn, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        flag = True
        for key, val in items:
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == 'leaveAt':
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == 'arriveBy':
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            return self.dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = delex.prepareSlotValuesIndependent()
        # self.delex_dialogues = json.load(open('resources/multi-woz-2.1/delex.json', 'r'))
        self.delex_dialogues = json.load(open('resources/multi-woz/delex.json', 'r'))

        self.db = MultiWozDB()
        self.labels = list()
        self.hyps = list()
        # self.venues = json.load(open('resources/all_venues.json', 'r'))

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
            # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                    # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _evaluateGeneratedDialogue(self, dialname, dial, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        m_targetutt = [turn['text'] for idx, turn in enumerate(realDialogue['log']) if idx % 2 == 1]

        # pred_beliefs = dial['aggregated_belief']
        pred_beliefs = dial['beliefs']
        target_beliefs = dial['target_beliefs']
        pred_responses = dial['responses']

        for t, (sent_gpt, sent_t) in enumerate(zip(pred_responses, m_targetutt)):

            for domain in goal.keys():

                if '[value_name]' in sent_gpt or '[value_id]' in sent_gpt:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION        
                        if domain not in pred_beliefs[t]:
                            venues = []
                        else: 
                            pred_beliefs = remove_model_mismatch_and_db_data(dialname, target_beliefs, pred_beliefs, domain, t)
                            venues = self.db.queryResultVenues(domain, pred_beliefs[t][domain], real_belief=True)

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = venues
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                # if ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if not flag and venues:  # sometimes there are no results so sample won't work
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                venue_offered[domain] = venues
                    else:  # not limited so we can provide one
                        # venue_offered[domain] = '[' + domain + '_name]'
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        # if domain + '_reference' in sent_t:
                        #     if 'restaurant_reference' in sent_t:
                        if '[value_reference]' in sent_gpt:
                            if domain == "restaurant" and realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            # elif 'hotel_reference' in sent_t:
                            elif domain == "hotel" and realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            # elif 'train_reference' in sent_t:
                            elif domain == "train" and realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        # if domain + '_' + requestable + ']' in sent_t:
                        if '[value_' + requestable + ']' in sent_gpt:
                            provided_requestables[domain].append(requestable)

            # print('venues', venue_offered)
            # print('request', provided_requestables)

        # if name was given in the task
        for domain in goal.keys():

            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)

                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                # elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if '[value_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match) / len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success) / len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats


    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, dialog['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                                    # return goal, 0, match, real_requestables
                            elif 'train_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            # if dialog['goal'][domain].has_key('info'):
            if 'info' in dialog['goal'][domain]:
                # if dialog['goal'][domain]['info'].has_key('name'):
                if 'name' in dialog['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        # HARD (0-1) EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match, success = 0, 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
                # print(goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1

            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, stats

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities

    def evaluateModel_gpt2(self, dialogues, real_dialogues=False, mode='valid'):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}


        for idx, (filename, dial) in tqdm(enumerate(dialogues.items())):

            filename = filename.upper() + ".json"

            data = delex_dialogues[filename]

            goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename)
            
            success, match, stats = self._evaluateGeneratedDialogue(filename, dial, goal,
                                                                         data, requestables,
                                                                         soft_acc=mode == 'soft')

            successes += success
            matches += match
            total += 1

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]

        if real_dialogues:
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for dialogue in dialogues:
                data = real_dialogues[dialogue]
                model_turns, corpus_turns = [], []
                # for idx, turn in enumerate(data['sys']):
                for idx, turn in enumerate(data):
                    corpus_turns.append([turn])
                for turn in dialogues[dialogue]['responses']:
                    model_turns.append([turn])

                # ipdb.set_trace()
                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise ('Wrong amount of turns')

            model_corpus_len = []
            for turn in model_corpus:
                if turn[0] == '':
                    model_corpus_len.append(True)
                else:
                    model_corpus_len.append(False)
            if all(model_corpus_len):
                print('no model response')
                model_corpus = corpus
            # ipdb.set_trace()
            blue_score = bscorer.score(model_corpus, corpus)
        else:
            blue_score = 0.

        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU : {:2.4f}'.format(mode, blue_score*100) + "\n"
        report += '{} Corpus Combined Score : {:2.4f}'.format(mode, ((matches / float(total) * 100) + (successes / float(total) * 100)) / 2 + blue_score*100) + "\n"
        report += 'Total number of dialogues: %s ' % total

        print(report)

        return report, successes / float(total), matches / float(total)


def postprocess_gpt2(generated_raw_data):
    generated_proc_data = {}
    for key, value in generated_raw_data.items():
        target_beliefs = value['target_turn_belief']
        target_beliefs_dict = []
        beliefs = value['generated_turn_belief']
        belief_dict = []

        for turn_bs in beliefs:
            bs_dict = {}
            for bs in turn_bs:
                if len(bs.split()) < 3:
                    continue
                if bs in ['', ' ']:
                    continue
                domain = bs.split()[0]
                if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                    print(key, domain)
                    continue
                if 'book' in bs:
                    continue
                slot = bs.split()[1]
                val = ' '.join(bs.split()[2:])
                if val == 'none':
                    continue
                if domain not in bs_dict:
                    bs_dict[domain] = {}
                bs_dict[domain][slot] = val
            belief_dict.append(bs_dict)

        aggregated_belief_dict = {}
        for bs in value['generated_belief']:
            if len(bs.split()) < 3:
                # print('skipping {}'.format(bs))
                continue
            domain = bs.split()[0]
            if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                print(domain)
                continue
            if 'book' in bs:
                continue
            slot = bs.split()[1]
            val = ' '.join(bs.split()[2:])
            if val == 'none':
                continue
            if domain not in aggregated_belief_dict:
                aggregated_belief_dict[domain] = {}
            aggregated_belief_dict[domain][slot] = val

        for turn_bs in target_beliefs:
            bs_dict = {}
            for bs in turn_bs:
                if bs in ['', ' ']:
                    continue
                domain = bs.split()[0]
                if domain not in ['train', 'taxi', 'hotel', 'hospital', 'attraction', 'restaurant']:
                    print(domain)
                    continue
                if 'book' in bs:
                    continue
                slot = bs.split()[1]
                val = ' '.join(bs.split()[2:])
                if val == 'none':
                    continue
                if domain not in bs_dict:
                    bs_dict[domain] = {}
                bs_dict[domain][slot] = val
            target_beliefs_dict.append(bs_dict)

        if aggregated_belief_dict != belief_dict[-1]:
            for domain in aggregated_belief_dict:
                if domain == 'attraction' and domain in belief_dict[-1] and len(
                        aggregated_belief_dict[domain].keys()) < len(belief_dict[-1][domain].keys()):
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]
                elif domain == 'restaurant' and domain in aggregated_belief_dict and 'name' in aggregated_belief_dict[
                    domain] and aggregated_belief_dict[domain]['name'] == 'lovell lodge' and domain in belief_dict[
                    -1] and 'name' in belief_dict[-1][domain] and belief_dict[-1][domain]['name'] == 'restaurant 17':
                    # ipdb.set_trace()
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]
                elif domain == 'restaurant' and 'name' in aggregated_belief_dict[domain] and \
                        aggregated_belief_dict[domain]['name'] == 'curry garden' and 'area' in aggregated_belief_dict[
                    domain] and aggregated_belief_dict[domain]['area'] == 'east' and domain in belief_dict:
                    aggregated_belief_dict[domain] = belief_dict[-1][domain]


        generated_proc_data[key] = {
            'name': key,
            'responses': value['generated_response'],
            'beliefs': belief_dict,
            'aggregated_belief': aggregated_belief_dict,
            'target_beliefs': target_beliefs_dict,
            'generated_action': value['generated_action'],
            'target_action': value['target_action'],
        }
    return generated_proc_data


if __name__ == '__main__':
    mode = "test"
    evaluator = MultiWozEvaluator(mode)

    eval_filename = sys.argv[1]
    data_dir = sys.argv[2]

    # with open("resources/test_dials_2.1_lexicalized.json", "r") as f:
    # with open("resources/test_dials_2.1.json", "r") as f:
    # with open("resources/test_dials.json", "r") as f:
    with open(os.path.join(data_dir, "test_dials.json"), "r") as f:
        human_raw_data = json.load(f)
    human_proc_data = {}
    for key, value in human_raw_data.items():
        human_proc_data[key] = value['sys']

    with open(eval_filename, "r") as f:
        generated_raw_data = json.load(f)

    generated_proc_data = postprocess_gpt2(generated_raw_data)

    # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
    generated_data = generated_proc_data

    evaluator.evaluateModel_gpt2(generated_data, human_proc_data, mode=mode)
