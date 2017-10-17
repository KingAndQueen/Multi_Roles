import os
import pdb
import numpy as np
class Vocab():
    def __init__(self,word2vec=None,embed_size=0):
        self.word2idx={'<eos>':0,'<go>':1,'<pad>':2,'<unk>':3}
        self.idx2word={0:'<eos>',1:'<go>',2:'<pad>',3:'<unk>'}
        self.embed_size=embed_size
    def add_vocab(self,words):
        if isinstance(words, (list, np.ndarray)):
            for word in words:
                if word not in self.word2idx:
                    index=len(self.word2idx)
                    self.word2idx[word]=index
                    self.idx2word[index]=word
        else:
            if words not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[words] = index
                self.idx2word[index] = words
    def word_to_index(self,word):
        self.add_vocab(word)
        return self.word2idx[word]
    def index_to_word(self,index):
        if index in self.idx2word:
            return self.idx2word[index]
        else:
            return '<unk>'
    @property
    def vocab_size(self):
        return len(self.idx2word)

def read_file(data_path,vocabulary,sentence_size,roles_number):
    f=open(data_path,'r')
    scene={}
    scenes = []
    last_speaker=''
    name_list=[]
    for lines in f:
        if len(lines)>2:
            name=lines[:lines.index(':')]
            #name_id=vocabulary.word_to_index(name)#for word in name.split()]
            sentence=lines[lines.index(':')+1:lines.index(':')+1+sentence_size-1] #sub 1 for eos
            sentence_id=[vocabulary.word_to_index(word)for word in sentence.split()]
            sentence_id.append(vocabulary.word_to_index('<eos>'))
            padding_len=max(sentence_size-len(sentence_id),0)
            for i in range(padding_len):
                sentence_id.append(vocabulary.word_to_index('<pad>'))
            scene[name]=sentence_id
            last_speaker=name
            if name in ['Monica','Joey','Chandler','Phoebe', 'Rachel','Ross']:
                name_list.append(vocabulary.word_to_index(name))
        else:
            if last_speaker not in scene:
                continue
            ans=scene[last_speaker]
            ans.pop()
            ans.insert(0,vocabulary.word_to_index('<go>'))
            scene['ans']=ans
            weight=[]
            for id in scene[last_speaker]:
                if id==vocabulary.word_to_index('<pad>'):
                    weight.append(0.0)
                else:
                    weight.append(1.0)
            scene['weight']=weight
            name_pad=max(roles_number-len(name_list),0)
            for i in range(name_pad):name_list.append(vocabulary.word_to_index('<pad>'))
            scene['name']=name_list
            scene[last_speaker]=sentence_size*[vocabulary.word_to_index('<pad>')]
            scenes.append(scene)
            scene={}
            name_list=[]
    f.close()
    return scenes

def get_data(data_path,vocabulary,sentence_size,roles_number):
    train_data_path = os.path.join(data_path, 'Test_processed.txt')
    test_data_path = os.path.join(data_path, 'Test_processed.txt')
    return read_file(train_data_path,vocabulary,sentence_size,roles_number),read_file(test_data_path,vocabulary,sentence_size,roles_number)