import os
import pdb
import pickle
# import nltk
import numpy as np
import codecs
import json
import nltk
import sklearn
import random

NAMELIST = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'others']
NAME_MAP_ID = {'Chandler': 0, 'Joey': 1, 'Monica': 2, 'Phoebe': 3, 'Rachel': 4, 'Ross': 5, 'others': 6, 'pad': 7}
ID_MAP_NAME={0:'Chandler',1:'Joey', 2:'Monica', 3:'Phoebe', 4:'Rachel', 5:'Ross',6:'others',7:'pad'}

class Vocab():
    def __init__(self, word2vec=None, embed_size=0):
        self.word2idx = {'<eos>': 0, '<go>': 1, '<pad>': 2, '<unk>': 3}
        self.idx2word = {0: '<eos>', 1: '<go>', 2: '<pad>', 3: '<unk>'}
        self.embed_size = embed_size

    def add_vocab(self, words):
        if isinstance(words, (list, np.ndarray)):
            for word in words:
                if word not in self.word2idx:
                    index = len(self.word2idx)
                    self.word2idx[word] = index
                    self.idx2word[index] = word
        else:
            if words not in self.word2idx:
                # print('adding new word',words)
                index = len(self.word2idx)
                self.word2idx[words] = index
                self.idx2word[index] = words

    def word_to_index(self, word):
        self.add_vocab(word)
        return self.word2idx[word]

    def index_to_word(self, index):
        if index in self.idx2word:
            return self.idx2word[index]
        else:
            return '<unk>'

    @property
    def vocab_size(self):
        return len(self.idx2word)


def read_file(data_path, vocabulary, sentence_size, roles_number, rl=False):
    global NAMELIST
    f = open(data_path, 'r')
    # f = open(data_path, 'r', encoding='utf-8', errors='surrogateescap e')
    # f = codecs.open(data_path, 'r', 'utf-8')
    scene = {}
    scenes = []
    last_speaker = ''
    name_list_ = []
    for lines in f:
        lines = lines.strip()[2:-5]
        if len(lines) > 2:
            name = lines[:lines.index(':')]
            if name not in ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']:
                name = 'others'
            # name_id=vocabulary.word_to_index(name)#for word in name.split()]
            sentence = lines[lines.index(':') + 1:]
            sentence = sentence.split()
            sentence_id = [vocabulary.word_to_index(word) for word in sentence]
            sentence_id = sentence_id[:sentence_size - 1]
            sentence_id.append(vocabulary.word_to_index('<eos>'))
            padding_len = max(sentence_size - len(sentence_id), 0)
            for i in range(padding_len):
                sentence_id.append(vocabulary.word_to_index('<pad>'))
            scene[name] = sentence_id
            last_speaker = name
            name_list_.append(name)
        else:

            if last_speaker not in scene or last_speaker == '':
                continue
            ans = scene[last_speaker]
            ans.pop()  # pop out the last word IN answer
            ans.insert(0, vocabulary.word_to_index('<go>'))  # padding <go>
            scene['answer'] = ans
            weight = []
            name_list = []
            for id in scene[last_speaker]:
                if id == vocabulary.word_to_index('<pad>'):
                    weight.append(0.0)
                else:
                    weight.append(1.0)
            scene['weight'] = weight
            # if not rl:
            #     name_list_.pop()  # pop the last speaker to hidden the true speaker for none-rl
            name_list_.pop()  # pop out the last speaker
            for name_ in NAMELIST:
                if name_ in name_list_:
                    name_list.append(NAME_MAP_ID[name_])
                else:
                    name_list.append(NAME_MAP_ID['pad'])
            # name_pad = roles_number - len(name_list)
            if len(name_list) != roles_number: pdb.set_trace()
            # for i in range(name_pad): name_list.append(vocabulary.word_to_index('<pad>'))
            scene['name_list'] = name_list
            scene[last_speaker] = sentence_size * [vocabulary.word_to_index('<pad>')]
            scene['speaker'] = NAME_MAP_ID[last_speaker]
            scenes.append(scene)
            scene = {}
            name_list_ = []
    f.close()
    return scenes


def get_data(data_path, vocabulary, sentence_size, roles_number, rl=False):
    train_data_path = os.path.join(data_path, 'train.txt')
    test_data_path = os.path.join(data_path, 'test.txt')
    valid_data_path = os.path.join(data_path, 'validation.txt')
    return read_file(train_data_path, vocabulary, sentence_size, roles_number), \
           read_file(valid_data_path, vocabulary, sentence_size, roles_number), \
           read_file(test_data_path, vocabulary, sentence_size, roles_number, rl)


def store_vocab(vocab, data_path):
    data_path = data_path + 'vocab.pkl'
    f = open(data_path, 'wb')
    pickle.dump(vocab, f)
    f.close()


def get_vocab(data_path):
    data_path = data_path + 'vocab.pkl'
    if os.path.exists(data_path):
        f = open(data_path, 'rb')
        vocab = pickle.load(f)
        f.close()
    else:
        print('<<<<<<< vocab is not exist >>>>>>>')
        vocab = None
    return vocab


def get_humorous_scene_rl(data_path, vocabulary, sentence_size):
    data_path = data_path + 'humorous_scenes.txt'

    if os.path.exists(data_path):
        f = open(data_path, 'r')
        name_list_ = []
        scene = {}
        scenes = []
        for lines in f:
            # pdb.set_trace()
            if len(lines) > 2:
                name = lines[:lines.index(':')]
                if name not in ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross']:
                    name = 'others'
                    # name_id=vocabulary.word_to_index(name)#for word in name.split()]
                sentence = lines[lines.index(':') + 1:]
                sentence = sentence.split()
                sentence_id = [vocabulary.word_to_index(word) for word in sentence]
                sentence_id = sentence_id[:sentence_size - 1]
                sentence_id.append(vocabulary.word_to_index('<eos>'))
                padding_len = max(sentence_size - len(sentence_id), 0)
                for i in range(padding_len):
                    sentence_id.append(vocabulary.word_to_index('<pad>'))
                scene[name] = sentence_id
                last_speaker = name
                name_list_.append(name)
            else:
                if last_speaker not in scene or last_speaker == '':
                    continue
                scene['answer'] = sentence_size * [vocabulary.word_to_index('<pad>')]
                name_list = []
                weight = sentence_size * [0]
                scene['weight'] = weight
                for name_ in NAMELIST:
                    if name_ in name_list_:
                        name_list.append(NAME_MAP_ID[name_])
                    else:
                        name_list.append(NAME_MAP_ID['pad'])
                scene['name'] = name_list
                scene['speaker'] =NAME_MAP_ID[last_speaker]
                scenes.append(scene)
                scene = {}
                name_list_ = []
        return scenes

    else:
        print('<<<<<<< humorous scenes is not exist >>>>>>>')


def read_tt_data(data_dir, vocabulary, sents_len):
    data_dir=data_dir+'/pre_train/'
    def structed(sents):
        sents_id = [vocabulary.word_to_index(word) for word in sents]
        sents_id = sents_id[:sents_len - 1]
        sents_id.append(vocabulary.word_to_index('<eos>'))
        padding_len = max(sents_len - len(sents_id), 0)
        for i in range(padding_len):
            sents_id.append(vocabulary.word_to_index('<pad>'))
        return sents_id

    if not os.path.exists(data_dir):
        print ('data_dir is not exist!')
        return None
    filelist = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            file_name = os.path.splitext(os.path.join(root, name))
            if file_name[1] == '.json':
                filelist.append(os.path.join(root, name))

    sents_data = []
    for datafile in filelist:
        f = open(datafile)
        line = f.readline()
        f.close()
        raw_data = json.loads(str(line.strip()))
        for data in raw_data:
            sents_q = nltk.word_tokenize(data['question'])
            sents_q = structed(sents_q)
            sents_data.append(sents_q)
            sents_a = nltk.word_tokenize(data['answer'])
            sents_a = structed(sents_a)
            sents_data.append(sents_a)
    scenes = []
    scene_null = {'Chandler': '', 'Joey': '', 'Monica': '', 'Phoebe': '', 'Rachel': '', 'Ross': '', 'others': '',
                  }
    while len(sents_data) > 8:
        scene = scene_null
        name_list_ = list(NAMELIST)
        for key, value in scene.items():
            if value == '':
                scene[key] = sents_data.pop()
        ans = sents_data.pop()
        ans.pop()
        ans.insert(0, vocabulary.word_to_index('<go>'))
        scene['answer'] = list(ans)
        weight = []
        for id in scene['answer']:
            if id == vocabulary.word_to_index('<pad>'):
                weight.append(0.0)
            else:
                weight.append(1.0)
            scene['weight'] = weight
        name_list = []
        random_speaker = random.randint(0, 6)
        speaker = name_list_.pop(random_speaker)
        scene[speaker]=sents_len * [vocabulary.word_to_index('<pad>')]
        for name_ in NAMELIST:
            if name_ in name_list_:
                name_list.append(NAME_MAP_ID[name_])
            else:
                name_list.append(NAME_MAP_ID['pad'])
        scene['name_list'] = name_list
        scene['speaker'] = NAME_MAP_ID[speaker]
        scenes.append(dict(scene))

    return scenes
