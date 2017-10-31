import random
import pdb

class Drama():
    def __init__(self, name):
        self.name = name
        self.script = []

    def reset(self, data_input_test):
        self.script = []
        test_sample = random.choice(data_input_test)
        self.script.append(test_sample)
        return test_sample

    def render(self):
        print(self.script)

    def check_state(self,vocab):
        def score(conversation):
            score_number = 0.0
            for sent in conversation.values():
                sent=list(sent[-1])
                if vocab.word_to_index('<eos>') in sent:
                    words = sent[:sent.index(vocab.word_to_index('<eos>'))]
                else:
                    words=sent
                score_number = len(words) - len(set(words))
            return score_number

        scores = 0.0
        for conversation in self.script:
            scores += score(conversation)
        if scores > 2:
            return True #too many useless words th
        else:
            return False

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
