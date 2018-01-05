import numpy as np
import tensorflow as tf
import pdb
import Multi_Roles_Model
import Multi_Roles_Data
# import copy
# reproducible
np.random.seed(1)
tf.set_random_seed(1)
NAMELIST = Multi_Roles_Data.NAMELIST
NAME_MAP_ID = Multi_Roles_Data.NAME_MAP_ID
ID_MAP_NAME=Multi_Roles_Data.ID_MAP_NAME


class PolicyGradient:
    def __init__(
            self,
            config,
            vocab,
            output_graph=False,
    ):
        self.config = config
        self.vocab = vocab
        self.lr = config.learn_rate
        self.gamma = config.reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.sess = tf.Session()
        self._build_net()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

    def _build_net(self):
        print('establish the model...')
        self.model = Multi_Roles_Model.MultiRolesModel(self.config, self.vocab)

        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoints_dir)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def choose_scene(self, obser):
        observation=obser
        observations = list()
        # pdb.set_trace()
        observations.append(dict(observation))
        name_list = list(observation['name_list'][-1])

        while NAME_MAP_ID['pad'] in name_list:
            name_list.remove(NAME_MAP_ID['pad'])
        role_number=len(name_list)
        # pdb.set_trace()
        for r in range(role_number):
            # pdb.set_trace()
            next_speaker = observation['speaker'][0]
            predict,new_speaker = self.choose_action(observation)
            observation['speaker'] = [new_speaker]
            del_speaker = name_list.pop(0)
            observation[ID_MAP_NAME[next_speaker]] = observation['answer']
            observation['answer'] = predict
            new_weight = []
            flag = False
            for ans in predict[-1]:
                if ans == self.vocab.word_to_index('<eos>'):
                    flag=True
                if ans != self.vocab.word_to_index('<pad>') and not flag:
                    new_weight.append(1.0)
                else:
                    new_weight.append(0.0)
            observation['weight'] = [new_weight]

            # update observation to construct new observation
            observation[ID_MAP_NAME[del_speaker]]=[len(predict[-1]) * [self.vocab.word_to_index('<pad>')]]
            observation[ID_MAP_NAME[new_speaker]] = [len(predict[-1]) * [self.vocab.word_to_index('<pad>')]]
            name_list.append(next_speaker)

            new_name_list = []
            for name in NAMELIST:
                name_id = NAME_MAP_ID[name]
                if name_id in name_list:
                    new_name_list.append(name_id)
                else:
                    new_name_list.append(NAME_MAP_ID['pad'])

            observation['name_list'] = [new_name_list]
            observations.append(dict(observation))  # save last observation for learning in RL

        return observations

    def choose_action(self, observation):
        loss, predict, _ ,next_speaker_vector= self.model.step(self.sess, observation, step_type='test')
        next_speaker=np.argmax(next_speaker_vector[0], 0)
        return predict,next_speaker

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)  # observations
        # self.ep_as.append(a)  # actions
        self.ep_rs.append(r)  # rewards

    def learn(self):
        # discount and normalize episode reward
        # discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        for index, conversation in enumerate(self.ep_obs):
            # conversation['answer'] = self.ep_as[index]
            # conversation['weight']=[]
            # if len(self.ep_rs) > 3:
            #     conversation['reward'] = discounted_ep_rs_norm[index]
            # pdb.set_trace()
            conversation['reward'] =self.ep_rs[index]
            loss= self.model.step(self.sess, conversation, step_type='rl_learn')

        # self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return loss

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        # pdb.set_trace()
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
