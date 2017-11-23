import os
import random

import numpy as np
import tensorflow as tf
import pdb
import Multi_Roles_Data
import Multi_Roles_Model

# set parameters of model
flags = tf.app.flags
flags.DEFINE_string('model_type', 'train', 'whether model initial from checkpoints')
flags.DEFINE_string('data_dir', 'data/', 'data path for model')
flags.DEFINE_string('checkpoints_dir', 'checkpoints/', 'path for save checkpoints')
flags.DEFINE_string('summary_path', './summary', 'path of summary for tensorboard')
flags.DEFINE_string('device_type', 'gpu', 'device for computing')

flags.DEFINE_boolean('rl', False, 'rl sign for model')

flags.DEFINE_integer('stop_limit', 5, 'number of evaluation loss is greater than train loss  ')
flags.DEFINE_integer('layers', 3, 'levels of rnn or cnn')
flags.DEFINE_integer('neurons', 100, 'neuron number of one level')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('roles_number', 7, 'number of roles in the data')
flags.DEFINE_integer('epoch', 1000, 'training times')
flags.DEFINE_integer('check_epoch', 100, 'training times')
flags.DEFINE_integer('sentence_size', 20, 'length of sentence')
flags.DEFINE_float('interpose', 0.5, 'value for gru gate to decide interpose')
flags.DEFINE_float('learn_rate', 0.01, 'value for gru gate to decide interpose')
flags.DEFINE_float("learning_rate_decay_factor", 1, 'if loss not decrease, multiple the lr with factor')
flags.DEFINE_float("max_grad_norm", 5, 'Clip gradients to this norm')

config = flags.FLAGS

def record_result(current_step=-1, eval_loss=-1, loss=-1):
    if not os.path.isfile('./result_data.txt'):
        result_file = open('./result_data.txt', 'a+')
        result_file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % ('layers',
                                                                        'neurons',
                                                                        'batch_size',
                                                                        'epoch',
                                                                        'interpose',
                                                                        'learn_rate',
                                                                        'learning_rate_decay_factor',
                                                                        'max_grad_norm',
                                                                        'stop_limit',
                                                                        'stop_step',
                                                                        'eval_loss',
                                                                        'train_loss',
                                                                        'checkpoints_dir'))
        result_file.flush()
    else:
        if current_step == -1:
            return
        result_file = open('./result_data.txt', 'a+')
        result_file.write("%d,%d,%d,%d,%f,%f,%f,%d,%d,%d,%d,%d,%s\n" % (config.layers,
                                                                  config.neurons,
                                                                  config.batch_size,
                                                                  config.epoch,
                                                                  config.interpose,
                                                                  config.learn_rate,
                                                                  config.learning_rate_decay_factor,
                                                                  config.max_grad_norm,
                                                                  config.stop_limit,
                                                                  current_step,
                                                                  eval_loss ,
                                                                  loss ,
                                                                  config.checkpoints_dir))
        result_file.flush()
    result_file.close()


def show_result(seq, vocab):
    if isinstance(seq, (list, np.ndarray)):
        words = []
        for idx in seq:
            if isinstance(idx, (list, np.ndarray)):
                show_result(idx, vocab)
            else:
                if vocab.index_to_word(idx) == '<eos>': break
                words.append(vocab.index_to_word(idx))
        print(words)
    if isinstance(seq, (str, int)):
        print(vocab.idx_to_word(seq))


def data_process(config, vocabulary=None):
    # read data from file and normalized
    if vocabulary == None:
        vocabulary = Multi_Roles_Data.Vocab()
    train_data, valid_data, test_data = Multi_Roles_Data.get_data(config.data_dir, vocabulary, config.sentence_size,
                                                                  config.roles_number,config.rl)
    print('data processed,vocab size:', vocabulary.vocab_size)
    Multi_Roles_Data.store_vocab(vocabulary, config.data_dir)
    return train_data, valid_data, test_data, vocabulary


# training model
def train_model(sess, model, train_data, valid_data):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 0
    data_input_train = model.get_batch(train_data)
    data_input_eval = model.get_batch(valid_data)

    train_summary_writer = tf.summary.FileWriter(config.summary_path + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(config.summary_path + '/test')
    print('training....')
    checkpoint_path = os.path.join(config.checkpoints_dir, 'MultiRoles.ckpt')
    train_losses = []
    eval_losses = []
    record_result()
    loss = float()
    eval_loss = float()
    while current_step < config.epoch:
        #  print ('current_step:',current_step)

        for i in range(len(data_input_train)):
            loss, _, summary_train = model.step(sess, random.choice(data_input_train))

        if current_step % config.check_epoch == 0:
            train_losses.append(loss)
            if len(train_losses) > 2 and loss > max(train_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training total loss:', loss)
            train_summary_writer.add_summary(summary_train, current_step)

            eval_data = random.choice(data_input_eval)
            eval_loss, _, summary_eval = model.step(sess, eval_data)
            eval_losses.append(eval_loss)
            test_summary_writer.add_summary(summary_eval)
            print('evaluation total loss:', eval_loss)
            print('saving current step %d checkpoints....' % current_step)
            model.saver.save(sess, checkpoint_path, global_step=current_step)
            if len(eval_losses) > config.stop_limit - 1 and eval_loss > max(eval_losses[-1 * config.stop_limit:]):
                print('----End training for evaluation increase----')
                break

        current_step += 1
    record_result(current_step, eval_loss, loss)


def test_model(sess, model, test_data, vocab):
    data_input_test = model.get_batch(test_data)
    loss_test = 0.0
    predicts = []
    f=open('')
    for batch_id, data_test in enumerate(data_input_test):
        loss, predict, _ ,vector= model.step(sess, data_test, step_type='test')
        loss_test += loss
        # pdb.set_trace()
        # print('labels: Id:', batch_id)
        # show_result(data_test.get('answer'), vocab)
        # predicts.append(predict)

        # print ('predicts: Id:', batch_id)
        # show_result(predict, vocab)
    print('test total loss:', loss_test / len(data_input_test))


    # data_input_test = model.get_batch(test_data)
    # test_sample=random.choice(data_input_test)
    # loss,predict,_=model.step(sess,test_sample,step_type='test')
    # show_result(test_sample.get('answer'),vocab)
    # show_result(predict,vocab)


# testing model
def main(_):
    train_data, valid_data, test_data, vocab = data_process(config)
    # initiall model from new parameters or checkpoints
    sess = tf.Session()
    print ('train data set %d, valid data set %d, test data set %d' %(len(train_data),len(valid_data),len(test_data)))
    if config.model_type == 'train':
        print('establish the model...')
        model = Multi_Roles_Model.MultiRolesModel(config, vocab)
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters....")
            sess.run(tf.global_variables_initializer())
        train_model(sess, model, train_data, valid_data)
        test_model(sess, model, test_data, vocab)
    if config.model_type == 'test':
        print('establish the model...')
        config.batch_size = len(test_data)
        model = Multi_Roles_Model.MultiRolesModel(config, vocab)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(sess, model, test_data, vocab)


if __name__ == "__main__":
    tf.app.run()
