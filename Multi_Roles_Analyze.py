# coding=utf8
import os
import pdb

import numpy as np


class Multi_Roles_Analyze():
    def __init__(self):
        pass

    def record_result(self, config, current_step=-1, eval_loss=-1, loss=-1, file_path='./result_data.txt'):
        if not os.path.isfile(file_path):
            result_file = open(file_path, 'a+')
            result_file.write(
                'layers,neurons,batch_size,epoch,interpose,'
                'learn_rate,learning_rate_decay_factor,max_grad_norm,'
                'stop_limit,stop_step,eval_loss,train_loss,eckpoints_dir')
            result_file.flush()
        else:
            if current_step == -1:
                return
            result_file = open('./result_data.txt', 'a+')
            result_file.write("%d,%d,%d,%d,%f,%f,%f,%d,%d,%d,%d,%d,%s\n"
                              % (config.layers, config.neurons, config.batch_size,
                                 config.epoch, config.interpose, config.learn_rate,
                                 config.learning_rate_decay_factor, config.max_grad_norm,
                                 config.stop_limit, current_step, eval_loss, loss, config.checkpoints_dir))
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

    def related_matrix(self, config, vocab, vectors, datas, data_type):
        name_dir = {'Chandler': 0, 'Joey': 1, 'Monica': 2, 'Phoebe': 3, 'Rachel': 4, 'Ross': 5, 'others': 6}
        relation_matrix = np.zeros(
            # NAMELIST = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'others']
            shape=(config.roles_number, config.roles_number),
            dtype=np.float64)
        if data_type == 1:  # real data 0/1
            for data in datas:
                name_set = set(vocab.idx2word[name_idx] for name_idx in data['name_list'] if name_idx != 7)
                speaker = vocab.idx2word[data['speaker']]
                # name_set.add(speaker)
                # print(name_set)
                for name in name_set:
                    relation_matrix[name_dir[str(name)]][name_dir[str(speaker)]] += 1
                    # pdb.set_trace()
        else:  # guess speaker
            for vector in vectors:
                for data in datas:
                    name_set = set(vocab.idx2word[name_idx] for name_idx in data['name_list'] if name_idx != 7)
                    speaker = vocab.idx2word[data['speaker']]
                    # name_set.add(speaker)
                    for name in name_set:
                        relation_matrix[name_dir[str(name)]:] += vector

        # pdb.set_trace()
        return relation_matrix  # testing model
