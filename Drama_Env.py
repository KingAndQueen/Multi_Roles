import random

class Drama():
    def __init__(self,name):
        self.name=name
        self.script=[]

    def reset(self,data_input_test):
        self.script=[]
        test_sample = random.choice(data_input_test)
        self.script.append(test_sample)
        return test_sample

    def render(self):
        print(self.script)

    def check_state(self):
        def score(sent):
            words=sent.split()
            score_number=len(words)-len(set(words))
            return score_number

        if score(self.script[-1])>2:
            return True
        else:
            return False

    def step(self,sentence):
        conversation=self.script[-1]
        speaker=conversation.get('name_list')[-1]
        conversation[speaker]=sentence
        self.script.append(conversation)
        observation_=self.script[-1]
        done=self.check_state()
        reward=len(self.script)
        return observation_,reward,done

def make(name):
    return Drama(name)