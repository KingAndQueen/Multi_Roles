import Drama_Env
from RL_Brain import PolicyGradient
from Multi_Roles_main import data_process
from Multi_Roles_Data import get_vocab,get_humorous_scene_rl
import tensorflow as tf
import pdb
flags=tf.app.flags
flags.DEFINE_float("reward_decay",0.9,'decay the reward of RL')
config=flags.FLAGS

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
config.batch_size=1
config.rl=True
#env.seed(1)     # reproducible, general Policy gradient has high variance
vocab=get_vocab(config.data_dir)
env = Drama_Env.make('Drama')
_,_,test_data,vocab,_=data_process(config,vocabulary=vocab)

RL_model = PolicyGradient(
    config,
    vocab,
    # output_graph=True,
)
data_input_test = RL_model.model.get_batch(test_data)

humor_data=get_humorous_scene_rl(config.data_dir,vocab,config.sentence_size)
humor_input_data=RL_model.model.get_batch(humor_data)

for i_episode in range(300):
    observation = env.reset(data_input_test)
    index=1
    while True:
        print('try ',index)
        index+=1
        if RENDER: env.render()
        scenes = RL_model.choose_scene(observation)
        observation_, reward, done = env.step_scene(scenes,vocab,RL_model,humor_input_data)
        RL_model.store_transition(observation_,None,reward)
        # RL.store_transition(observation, sentence, reward)
        print (observation_)
        if done:
            print('------positive sentence!----')
            ep_rs_sum = sum(RL_model.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL_model.learn()

            break

        observation = observation_