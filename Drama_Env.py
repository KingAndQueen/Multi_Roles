class Drama():
    def __init__(self,name):
        self.name=name
        self.script=[]

    def reset(self):
        self.script=[]
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
        self.script.append(sentence)
        observation_=self.script
        done=self.check_state()
        reward=len(self.script)
        return observation_,reward,done

def make(name):
    return Drama(name)