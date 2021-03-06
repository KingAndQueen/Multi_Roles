import Drama_Env
from RL_Brain import PolicyGradient
from Multi_Roles_main import data_process
from Multi_Roles_Data import get_vocab,get_humorous_scene_rl
import Multi_Roles_Analyze
import tensorflow as tf
import pdb
flags=tf.app.flags
flags.DEFINE_float("reward_decay",0.9,'decay the reward of RL')
config=flags.FLAGS

DISPLAY_REWARD_THRESHOLD = 10  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
config.batch_size=1
config.rl=True
config.model_type='test'
config.checkpoints_dir='checkpoints/rl/'
#env.seed(1)     # reproducible, general Policy gradient has high variance
vocab=get_vocab(config.data_dir)
env = Drama_Env.make('Drama')
_,_,test_data,vocab,_=data_process(config,vocabulary=vocab)
# pdb.set_trace()
RL_model = PolicyGradient(
    config,
    vocab,
    # output_graph=True,
)
data_input_test = RL_model.model.get_batch(test_data)

humor_data=get_humorous_scene_rl(config.data_dir,vocab,config.sentence_size)
humor_input_data=RL_model.model.get_batch(humor_data)
analyze = Multi_Roles_Analyze.Multi_Roles_Analyze(config)
for i_episode in range(30):
    observation = env.reset(data_input_test)
    index=1
    while True:
        print('try:',index)
        # if RENDER: env.render()
        pdb.set_trace()
        print('index:',index)
        scenes = RL_model.choose_scene(observation)
        observation_, reward, done = env.step_scene(scenes,vocab,RL_model,humor_input_data)
        RL_model.store_transition(observation_,None,reward)
        # RL.store_transition(observation, sentence, reward)
        for scene in scenes:
            analyze.show_only_scene(scene,vocab)
            print ('--------------')
        # print (observation_)
        # pdb.set_trace()
        if done:
            ep_rs_last = RL_model.ep_rs[-1]

            if 'running_reward' not in globals():
                running_reward = ep_rs_last
            else:
                running_reward = (running_reward-ep_rs_last) * 0.99 + ep_rs_last * 0.01
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", running_reward)
            pdb.set_trace()
            vt = RL_model.learn()
            print('------positive sentence!----')
            # if running_reward >0.0016:
            #     break
        index += 1
        observation = observation_