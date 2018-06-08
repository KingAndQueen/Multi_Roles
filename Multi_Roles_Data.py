import os
import pdb
import pickle
import numpy as np
import json
import random
import nltk
import re
NAMELIST = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross', 'others']
NAME_MAP_ID = {'chandler': 0, 'joey': 1, 'monica': 2, 'phoebe': 3, 'rachel': 4, 'ross': 5, 'others': 6, 'pad': 7}
ID_MAP_NAME={0:'chandler',1:'joey', 2:'monica', 3:'phoebe', 4:'rachel', 5:'ross',6:'others',7:'pad'}
unlegal='[^A-Za-z0-9\ \']'
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
        # for rl
        # if word in self.word2idx:
        #     return self.word2idx[word]
        # else:
        #     return self.word2idx['<unk>']

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
    questioner=''
    name_list_ = []
    for lines in f:
        # lines = lines.strip()[2:-5]
        if len(lines) > 2:
            name = lines[:lines.index(':')].lower()
            if name not in ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']:
                name = 'others'
            # name_id=vocabulary.word_to_index(name)#for word in name.split()]
            sentence = lines[lines.index(':') + 1:]
            sentence = re.sub(unlegal, ' ', sentence)
            sentence = sentence.lower()
            sentence=nltk.word_tokenize(sentence)
            if len(sentence)<=0 or cmp(sentence[0],'idt')==0:
                # print(lines)
                # pdb.set_trace()
                continue
            # sentence = sentence.split()
            sentence_id = [vocabulary.word_to_index(word) for word in sentence]
            sentence_id = sentence_id[:sentence_size - 1]
            sentence_id.append(vocabulary.word_to_index('<eos>'))
            padding_len = max(sentence_size - len(sentence_id), 0)
            for i in range(padding_len):
                sentence_id.append(vocabulary.word_to_index('<pad>'))
            scene[name] = sentence_id
            questioner= last_speaker
            last_speaker = name
            name_list_.append(name)
            if len(name_list_)>len(NAMELIST):
                name_list_.pop(0)
                    # pdb.set_trace()
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
            _last_speaker=name_list_.pop()  # pop out the last speaker
            if not cmp(_last_speaker,last_speaker)==0:
                pdb.set_trace()

            # for name_ in NAMELIST:
            #     if name_ in name_list_:
            #         name_list.append(NAME_MAP_ID[name_])
            #     else:
            #         name_list.append(NAME_MAP_ID['pad'])

            for name_ in name_list_: # have the better result than previous 5 lines
                if name_ not in NAMELIST: pdb.set_trace()
                name_list.append(NAME_MAP_ID[name_])
            pad_numb=max(len(NAMELIST)-len(name_list),0)
            for x in range(pad_numb):
                name_list.append(NAME_MAP_ID['pad'])

            # name_pad = roles_number - len(name_list)
            if len(name_list) != roles_number: pdb.set_trace()
            # for i in range(name_pad): name_list.append(vocabulary.word_to_index('<pad>'))
            scene['name_list'] = name_list
            scene[last_speaker] = sentence_size * [vocabulary.word_to_index('<pad>')]
            scene['speaker'] = NAME_MAP_ID[last_speaker]
            scene['question']=scene[questioner]
            scenes.append(scene)
            # pdb.set_trace()
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
    data_path = data_path + 'humorous.txt'

    if os.path.exists(data_path):
        f = open(data_path, 'r')
        name_list_ = []
        scene = {}
        scenes = []
        questioner = ''
        last_speaker = ''
        for lines in f:
            #lines = lines.strip()[2:-5]
            # pdb.set_trace()
            if len(lines) > 2:

                name = lines[:lines.index(':')].lower()
                if name not in ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']:
                    name = 'others'
                    # name_id=vocabulary.word_to_index(name)#for word in name.split()]
                sentence = lines[lines.index(':') + 1:]
                sentence=sentence.lower()
                sentence = nltk.word_tokenize(sentence)
                # sentence = sentence.split()
                sentence_id = [vocabulary.word_to_index(word) for word in sentence]
                sentence_id = sentence_id[:sentence_size - 1]
                sentence_id.append(vocabulary.word_to_index('<eos>'))
                padding_len = max(sentence_size - len(sentence_id), 0)
                for i in range(padding_len):
                    sentence_id.append(vocabulary.word_to_index('<pad>'))
                scene[name] = sentence_id
                questioner = last_speaker
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
                scene['name_list'] = name_list
                scene['speaker'] =NAME_MAP_ID[last_speaker]
                scene['question'] = scene[questioner]
                scenes.append(scene)
                scene = {}
                name_list_ = []
        return scenes

    else:
        print('<<<<<<< humorous scenes is not exist >>>>>>>')


def read_tt_data(data_dir, vocabulary, sents_len):
    data_dir=data_dir+'/tick_tock/'
    def structed(sents):
        sents_id = [vocabulary.word_to_index(word) for word in sents]
        sents_id = sents_id[:sents_len - 1]
        sents_id.append(vocabulary.word_to_index('<eos>'))
        padding_len = max(sents_len - len(sents_id), 0)
        for i in range(padding_len):
            sents_id.append(vocabulary.word_to_index('<pad>'))
        return sents_id

    if not os.path.exists(data_dir):
        print ('ticktock data_dir is not exist!')
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
            sents_q = nltk.word_tokenize(data['question'].lower())
            # sents_q = data['question'].split()
            sents_q = structed(sents_q)
            sents_data.append(sents_q)
            sents_a = nltk.word_tokenize(data['answer'].lower())
            # sents_a = data['answer'].split()
            sents_a = structed(sents_a)
            sents_data.append(sents_a)
    scenes = []
    scene_null = {'chandler': '', 'joey': '', 'monica': '', 'phoebe': '', 'rachel': '', 'ross': '', 'others': '',
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
def get_role_test_data(test_data):
    # pdb.set_trace()
    sents_last = []
    for id ,name in enumerate(NAMELIST):
        if id==0:
            sents_last = test_data[name]
        if id<len(NAMELIST)-1:
            sents=list(test_data[NAMELIST[id+1]])
            test_data[NAMELIST[id+1]]=sents_last
            sents_last=sents
        else:
            test_data[NAMELIST[0]]=sents_last
    name_list=test_data['name_list'][0]
    name_list_simple=[]
    for name in name_list:
        if name==NAME_MAP_ID['pad']:
            name_list_simple.append(0)
        else:
            name_list_simple.append(1)
    last_simple=name_list_simple.pop()
    name_list_simple.insert(0,last_simple)
    new_name_list=[]
    for id,name_s in enumerate(name_list_simple):
        if name_s==1:
            new_name_list.append(NAME_MAP_ID[NAMELIST[id]])
        else:
            new_name_list.append(NAME_MAP_ID['pad'])
    test_data['name_list']=[new_name_list]
    return test_data


def read_twitter_data(data_dir, vocabulary, sents_len):
    data_dir=data_dir+'/twitter/'
    def structed(sents):
        sents_id = [vocabulary.word_to_index(word) for word in sents]
        sents_id = sents_id[:sents_len - 1]
        sents_id.append(vocabulary.word_to_index('<eos>'))
        padding_len = max(sents_len - len(sents_id), 0)
        for i in range(padding_len):
            sents_id.append(vocabulary.word_to_index('<pad>'))
        return sents_id

    if not os.path.exists(data_dir):
        print ('twitter data_dir is not exist!')
        return None
    filelist = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            file_name = os.path.splitext(os.path.join(root, name))
            if file_name[1] == '.txt':
                filelist.append(os.path.join(root, name))

    sents_data = []
    for datafile in filelist:
        f = open(datafile)
        for data in f:
            sents_new = nltk.word_tokenize(data.lower())
            # sents_q = data['question'].split()
            sents_new = structed(sents_new)
            sents_data.append(sents_new)
        f.close()
    # pdb.set_trace()
    scenes = []
    scene_null = {'chandler': '', 'joey': '', 'monica': '', 'phoebe': '', 'rachel': '', 'ross': '', 'others': '',
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