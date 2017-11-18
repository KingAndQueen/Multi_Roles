import Drama_Env
from RL_Brain import PolicyGradient
from Multi_Roles_main import data_process
from Multi_Roles_Data import get_vocab
import tensorflow as tf

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
_,_,test_data,vocab=data_process(config,vocabulary=vocab)

RL = PolicyGradient(
    config,
    vocab,
    # output_graph=True,
)

data_input_test = RL.model.get_batch(test_data)


for i_episode in range(300):
    observation = env.reset(data_input_test)
    index=1
    while True:
        print('try ',index)
        index+=1
        if RENDER: env.render()
        sentence = RL.choose_scene(observation)
        observation_, reward, done = env.step(sentence,vocab)
        RL.store_transition(observation, sentence, reward)
        print (sentence)
        if done:
            print('------positive sentence!----')
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()
            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            break

        observation = observation_