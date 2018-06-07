# coding=utf8
import os
import pdb
import Multi_Roles_Data
import numpy as np
import collections
import math

NAME_MAP_ID = Multi_Roles_Data.NAME_MAP_ID
ID_MAP_NAME = Multi_Roles_Data.ID_MAP_NAME

NAMELIST = list(Multi_Roles_Data.NAMELIST)
NAMELIST.append('answer')


class Multi_Roles_Analyze():
    def __init__(self, config):
        self.relation_matrix = np.zeros(shape=(config.roles_number, config.roles_number),
                                        dtype=np.float64)

    def show_only_scene(self, scene, vocab):
        for name in NAMELIST:
            sents_id = scene[name][0]
            sents = [vocab.index_to_word(id) for id in sents_id]

            while '<pad>' in sents:
                sents.pop(sents.index('<pad>'))
            if '<eos>' in sents:
                sents = sents[:sents.index('<eos>')]
            print(name, sents)

    def show_role_scene(self, predict, data_input_test_sample, vocab):
        # assert len(predict) == len(NAMELIST), 'role length is not correct'
        batch_size = len(data_input_test_sample[NAMELIST[0]])
        for id in range(batch_size):
            print('------')
            for name in NAMELIST:
                sents_id = data_input_test_sample[name]
                sents = [vocab.index_to_word(idx) for idx in sents_id[id]]
                while '<pad>' in sents:
                    sents.pop(sents.index('<pad>'))
                if '<eos>' in sents:
                    sents = sents[:sents.index('<eos>')]
                print(name, sents)
            for no, pred in enumerate(predict):
                pred_id = pred[id]
                pred_vale = [vocab.index_to_word(idx) for idx in pred_id]
                print ('pred %d : % r' % (no, pred_vale))

    def show_scene(self, predict_, data_test_, vocab):
        assert len(predict_) == len(data_test_), 'data length is not correct'
        data_all = zip(predict_, data_test_)
        for predict, data_test in data_all:
            for key, value_id in enumerate(predict):
                # print('--------------------')
                for name in NAMELIST:
                    sents_id = data_test[name]
                    sents = [vocab.index_to_word(id) for id in sents_id[key]]

                    while '<pad>' in sents:
                        sents.pop(sents.index('<pad>'))
                    if '<eos>' in sents:
                        sents = sents[:sents.index('<eos>')]
                    print(name, sents)
                # pdb.set_trace()
                true_speaker=data_test['speaker'][key]
                print('ture speaker',ID_MAP_NAME[true_speaker])
                value = [vocab.index_to_word(id) for id in value_id]
                print('predict:%s' % value)
                print('--------------------')

    def record_result(self, config, current_step=-1, train_loss=-1, eval_loss=-1, file_path='./result_data.txt'):
        if not os.path.isfile(file_path):
            result_file = open(file_path, 'a+')
            result_file.write(
                'layers,neurons,batch_size,epoch,interpose,'
                'learn_rate,learning_rate_decay_factor,max_grad_norm,'
                'stop_limit,stop_step,train_loss,eval_loss,eckpoints_dir')
            result_file.flush()

        if current_step == -1:
                return
        result_file = open('./result_data.txt', 'a+')
        result_file.write("%d,%d,%d,%d,%f,%f,%f,%d,%d,%d,%d,%d,%s\n"
                              % (config.layers, config.neurons, config.batch_size,
                                 config.epoch, config.interpose, config.learn_rate,
                                 config.learning_rate_decay_factor, config.max_grad_norm,
                                 config.stop_limit, current_step, train_loss, eval_loss, config.checkpoints_dir))
        result_file.flush()
        result_file.close()

    def show_result(self, seq, vocab):
        if isinstance(seq, (list, np.ndarray)):
            words = []
            for idx in seq:
                if isinstance(idx, (list, np.ndarray)):
                    self.show_result(idx, vocab)
                else:
                    if vocab.index_to_word(idx) == '<eos>': break
                    words.append(vocab.index_to_word(idx))
            print(words)
        if isinstance(seq, (str, int)):
            print(vocab.idx_to_word(seq))

    def related_matrix(self, vectors_batch, datas_batch, data_type):
        name_dir = {'chandler': 0, 'joey': 1, 'monica': 2, 'phoebe': 3, 'rachel': 4, 'ross': 5, 'others': 6}
        # relation_matrix = np.zeros(
        #     # NAMELIST = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'others']
        #     shape=(config.roles_number, config.roles_number),
        #     dtype=np.float64)
        assert len(vectors_batch) == len(datas_batch), 'input matrix length is wrong'
        data_all = zip(vectors_batch, datas_batch)
        right = 0.0
        for vectors, datas in data_all:
            if data_type == 1:  # real data 0/1
                for id, name_list in enumerate(datas['name_list']):
                    name_set = set(ID_MAP_NAME[name_idx] for name_idx in name_list if name_idx != 7)
                    speaker = ID_MAP_NAME[datas['speaker'][id]]
                    # name_set.add(speaker)
                    # print(name_set)
                    for name in name_set:
                        self.relation_matrix[name_dir[str(name)]][name_dir[str(speaker)]] += 1
                        # pdb.set_trace()
            else:  # guess speaker

                for id, name_list in enumerate(datas['name_list']):
                    name_set = set(ID_MAP_NAME[name_idx] for name_idx in name_list if name_idx != 7)
                    true_speaker = datas['speaker'][id]
                    # name_set.add(speaker)
                    if true_speaker == np.argmax(vectors[id], axis=0):
                        right += 1
                    for name in name_set:
                        # relation_matrix[name_dir[str(name)]:] += vector
                        self.relation_matrix[name_dir[str(name)]][np.argmax(vectors[id], axis=0)] += 1
                        # pdb.set_trace()
        acc = float(right) / (len(datas_batch) * len(datas['speaker']))
        print('next speaker accuracy:', acc)
        return self.relation_matrix  # testing model

    def _get_ngrams(self,segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.
        Returns:
          The Counter containing all n-grams upto max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu(self,reference_corpus, translation_corpus, max_order=4,
                     smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
          reference_corpus: list of lists of references for each translation. Each
              reference should be tokenized into a list of tokens.
          translation_corpus: list of translations to score. Each translation
              should be tokenized into a list of tokens.
          max_order: Maximum n-gram order to use when computing BLEU score.
          smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
          3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
          precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(reference_corpus,
                                             translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)
            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                 (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                     possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return (bleu, precisions, bp, ratio, translation_length, reference_length)