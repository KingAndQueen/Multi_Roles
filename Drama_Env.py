import random
import pdb
import math
import nltk
import numpy as np
class Drama():
    def __init__(self, name):
        self.name = name
        self.script = []

    def reset(self, data_input_test):
        self.script = []
        test_sample = random.choice(data_input_test)
        pdb.set_trace()
        self.script.append(test_sample)
        return test_sample

    def render(self):
        print(self.script)

    def check_state(self,vocab):
        def score(conversation):

            speaker_list=conversation.get('name_list')[-1]
            speaker=''
            for spk in reversed(speaker_list):
                if spk!=vocab.word_to_index('<pad>'):
                    speaker=spk
            sent=conversation.get(vocab.index_to_word(speaker))
            # pdb.set_trace()
            sent=(sent[-1])
            if vocab.word_to_index('<eos>') in sent:
                words = sent[:sent.index(vocab.word_to_index('<eos>'))]
            else:
                words=sent
            score_number1 = len(words)-(len(words) - len(set(words)))
            score_=[]
            for key,value in conversation.iteritems():
                if key !='name_list' and key !='weight' and key!='answer':
                    # pdb.set_trace()
                    bleu =nltk.translate.bleu(value, words)
                    score_.append(bleu)
            score_number2=max(score_)
            pdb.set_trace()
            score_number=score_number2+score_number1
            # pdb.set_trace()
            return score_number

        scores = 0.0
        for conversation in self.script:
            scores += score(conversation)
        if scores < 2*len(self.script):
            return False
        else:
            return True

    def step(self, sentence, vocab):
        conversation = self.script[-1]
        name_list = conversation.get('name_list')
        speaker=''
        for name in reversed(name_list[-1]):
            if name!=vocab.word_to_index('<pad>'):
                speaker = name
                break
        if speaker=='': pdb.set_trace()
        speaker_name = vocab.index_to_word(speaker)
        conversation[speaker_name] = sentence
        self.script.append(conversation)
        observation_ = self.script[-1]

        done = self.check_state(vocab)
        reward = float(len(self.script))
        return observation_, reward, done


def make(name):
    return Drama(name)
