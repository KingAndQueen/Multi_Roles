import Drama_Env
from RL_Brain import PolicyGradient
from Multi_Roles_main import data_process
from Multi_Roles_Data import get_vocab
import matplotlib.pyplot as plt
import tensorflow as tf

flags=tf.app.flags
# flags.DEFINE_string('model_type','train','whether model initial from checkpoints')
# flags.DEFINE_string('data_dir','data/','data path for model')
# flags.DEFINE_string('checkpoints_dir','checkpoints/','path for save checkpoints')
# flags.DEFINE_string('summary_path','./summary','path of summary for tensorboard')
# flags.DEFINE_string('device_type','gpu','device for computing')
#
# flags.DEFINE_integer('layers',1,'levels of rnn or cnn')
# flags.DEFINE_integer('neurons',50,'neuron number of one level')
# flags.DEFINE_integer('batch_size',1, 'batch_size should be 1 in RL process')
# flags.DEFINE_integer('roles_number',6,'number of roles in the data')
# flags.DEFINE_integer('epoch',10,'training times' )
# flags.DEFINE_integer('check_epoch',5,'training times' )
# flags.DEFINE_integer('sentence_size',20,'length of sentence')
# flags.DEFINE_float('interpose',0.5,'value for gru gate to decide interpose')
# flags.DEFINE_float('learn_rate',0.5,'value for gru gate to decide interpose')
# flags.DEFINE_float("learning_rate_decay_factor", 0.99,'if loss not decrease, multiple the lr with factor')
# flags.DEFINE_float("max_grad_norm",5,'Clip gradients to this norm')
flags.DEFINE_float("reward_decay",0.9,'decay the reward of RL')
config=flags.FLAGS



DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
config.batch_size=1
#env.seed(1)     # reproducible, general Policy gradient has high variance
vocab=get_vocab(config.data_dir)
env = Drama_Env.make('Drama')
_,test_data,vocab=data_process(config,vocabulary=vocab)

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
        sentence = RL.choose_action(observation)
        observation_, reward, done = env.step(sentence,vocab)
        RL.store_transition(observation, sentence, reward)

        if done:
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