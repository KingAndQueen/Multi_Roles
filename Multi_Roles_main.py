import os
import random
import numpy as np
import tensorflow as tf
import pdb
import pickle as pkl
import Multi_Roles_Analyze
import Multi_Roles_Data
import Multi_Roles_Model
from math import exp
from sklearn import model_selection
ID_MAP_NAME=Multi_Roles_Data.ID_MAP_NAME
NAMELIST=Multi_Roles_Data.NAMELIST
# set parameters of model
flags = tf.app.flags
flags.DEFINE_string('model_type', 'train', 'whether model initial from checkpoints')
flags.DEFINE_string('data_dir', 'data/', 'data path for model')
flags.DEFINE_string('checkpoints_dir', 'checkpoints', 'path for save checkpoints')
flags.DEFINE_string('summary_path', './summary', 'path of summary for tensorboard')
flags.DEFINE_string('device_type', 'gpu', 'device for computing')

flags.DEFINE_boolean('rl', False, 'rl sign for model')

flags.DEFINE_integer('stop_limit', 5, 'number of evaluation loss is greater than train loss  ')
flags.DEFINE_integer('layers', 3, 'levels of rnn or cnn')
flags.DEFINE_integer('neurons', 800, 'neuron number of one level')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('roles_number', 7, 'number of roles in the data')
flags.DEFINE_integer('epoch', 2000, 'training times')
flags.DEFINE_integer('check_epoch',200, 'evaluation times')
flags.DEFINE_integer('sentence_size', 20, 'length of sentence')
flags.DEFINE_float('interpose', 0.5, 'value for gru gate to decide interpose')
flags.DEFINE_float('learn_rate', 0.1, 'value for gru gate to decide interpose')
flags.DEFINE_float("learning_rate_decay_factor", 0.5, 'if loss not decrease, multiple the lr with factor')
flags.DEFINE_float("max_grad_norm", 5, 'Clip gradients to this norm')

config = flags.FLAGS


def data_process(config, vocabulary=None):
    # read data from file and normalized
    if vocabulary == None:
        vocabulary = Multi_Roles_Data.Vocab()
    train_data, valid_data, test_data = Multi_Roles_Data.get_data(config.data_dir, vocabulary, config.sentence_size,
                                                                  config.roles_number, config.rl)
    pre_train_data = Multi_Roles_Data.read_tt_data(config.data_dir, vocabulary, config.sentence_size)
    # pre_train_data =[]
    Multi_Roles_Data.get_humorous_scene_rl(config.data_dir, vocabulary, config.sentence_size)
    print('data processed,vocab size:', vocabulary.vocab_size)
    Multi_Roles_Data.store_vocab(vocabulary, config.data_dir)
    all_data = train_data + valid_data + test_data
    train_data, eval_test_data = model_selection.train_test_split(all_data, test_size=0.2)
    valid_data, test_data = model_selection.train_test_split(eval_test_data, test_size=0.5)

    return train_data, valid_data, test_data, vocabulary, pre_train_data


# training model
def train_model(sess, model, analyze, train_data, valid_data, pretrain_epoch=0):
    # train_data, eval_data = model_selection.train_test_split(train_data, test_size=0.2)
    current_step = 1
    data_input_train = model.get_batch(train_data)
    data_input_eval = model.get_batch(valid_data)

    train_summary_writer = tf.summary.FileWriter(config.summary_path + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(config.summary_path + '/test')

    checkpoint_path = os.path.join(config.checkpoints_dir, 'MultiRoles.ckpt')
    train_losses = []

    analyze.record_result(config)
    loss = float()
    eval_loss = float()
    if pretrain_epoch>0:
        epoch=pretrain_epoch
        print('pre-training....')
    else:
        epoch=config.epoch
        print('training....')
    eval_losses_all = []
    while current_step <= epoch:
        #  print ('current_step:',current_step)

        for i in range(len(data_input_train)):
            train_loss_, _, summary_train = model.step(sess, random.choice(data_input_train),step_type='train')

        if current_step % config.check_epoch == 0:
            eval_losses = []
            train_losses.append(train_loss_)
            # if len(train_losses) > 2 and train_loss_ > max(train_losses[-3:]):
            #     print('decay learning rate....')
            #     sess.run(model.learning_rate_decay_op)
            print('-------------------------------')
            print('current_step:', current_step)
            print('training loss:', train_loss_)
            train_summary_writer.add_summary(summary_train, current_step)

            # eval_data = random.choice(data_input_eval)
            for eval_data in data_input_eval:
                eval_loss_, _, summary_eval = model.step(sess, eval_data)
                eval_losses.append(eval_loss_)
            test_summary_writer.add_summary(summary_eval)
            eval_loss=float(sum(eval_losses)) / len(eval_losses)

            print('evaluation loss:', eval_loss)
            if eval_loss<300 and train_loss_ <300:
                print('train perplex:',exp(train_loss_))
                print('evaluation perplex:',exp(eval_loss))
            print('saving current step %d checkpoints....' % current_step)
            model.saver.save(sess, checkpoint_path, global_step=current_step)
            if len(eval_losses_all) > 2 and eval_loss > eval_losses_all[-1]:
                print('decay learning rate....')
                sess.run(model.learning_rate_decay_op)
            if len(eval_losses_all) > config.stop_limit - 1 and eval_loss > max(eval_losses_all[-1 * config.stop_limit:]):
                print('----End training for evaluation increase----')
                break
            eval_losses_all.append(eval_loss)
        current_step += 1
        analyze.record_result(config, current_step, eval_loss, loss)


def test_model(sess, model, analyze, test_data, vocab):
    data_input_test = model.get_batch(test_data)
    test_loss = 0.0
    predicts = []
    vectors=[]
    for batch_id, data_test in enumerate(data_input_test):
        loss, predict, _, vector = model.step(sess, data_test, step_type='test')
        test_loss += loss
        predicts.append(predict)
        vectors.append(vector)

    print(analyze.related_matrix(vectors, data_input_test, 0))
    analyze.show_scene(predicts,data_input_test,vocab)
    test_loss=test_loss / len(data_input_test)
    print('test total loss:', test_loss)
    if test_loss<300:
        print ('test perplex:',exp(test_loss))


def test_role_model(sess, model, analyze, test_data, vocab) :   #test role defined generator
    print('test role defined generator')
    # pdb.set_trace()
    data_input_test= model.get_batch(test_data)
    # data_input_test_role= random.choice(data_input_test)
    for data_input_test_role in data_input_test:
        _, predict, _, vector = model.step(sess, data_input_test_role, step_type='test')
        role_predict=np.argmax(vector[0],0)
        print('predict speaker:',ID_MAP_NAME[role_predict])
        analyze.show_scene([predict], [data_input_test_role], vocab)
        data_input_test_sample=Multi_Roles_Data.get_role_test_data(data_input_test_role)
        _,predict,_, vector = model.step(sess, data_input_test_sample, step_type='test')
        role_predict = np.argmax(vector[0], 0)
        print('predict speaker:', ID_MAP_NAME[role_predict])
        analyze.show_scene([predict], [data_input_test_sample], vocab)
        print('#############################')


def test_online(sess, model, test_data, vocab,config):
    for key, value in test_data:
        sents=value.split()
        sentence_id=[vocab.word_to_index(word) for word in sents]
        sentence_id = sentence_id[:config.sentence_size]
        padding_len = max(config.sentence_size - len(sentence_id), 0)
        for i in range(padding_len):
            sentence_id.append(vocab.word_to_index('<pad>'))
        test_data[key]=sentence_id
    name_list=[]
    for role in NAMELIST:
        if role in test_data.keys():
            name_list.append(1)
        else:
            name_list.append(0)
    test_data['name_list']=name_list
    test_data['answer']=[config.sentence_size*[vocab.word_to_index('<pad>')]]
    test_data['weight'] = [config.sentence_size * [vocab.word_to_index('<pad>')]]
    test_data['speaker'] = [vocab.word_to_index('<pad>')]
    data_input_test = model.get_batch(test_data)
    _, predict, _, _ = model.step(sess, data_input_test, step_type='test')
    predict = [vocab.index_to_word(id) for id in predict]
    return  predict

# testing model
def main(_):
    train_data, valid_data, test_data, vocab, pre_train_data = data_process(config)
    # initiall model from new parameters or checkpoints
    sess = tf.Session()
    print('pretrain data set %d, train data set %d, valid data set %d, test data set %d' % (
        len(pre_train_data), len(train_data), len(valid_data), len(test_data)))
    # my_embedding = pkl.load(open('my_embedding.pkl', 'rb'))
    if config.model_type == 'train':
        print('establish the model...')
        model = Multi_Roles_Model.MultiRolesModel(config, vocab)
        # model = Multi_Roles_Model.MultiRolesModel(config, vocab, my_embedding=my_embedding)
        analyze = Multi_Roles_Analyze.Multi_Roles_Analyze(config)
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters....")
            sess.run(tf.global_variables_initializer())
        train_model(sess, model, analyze, pre_train_data, valid_data,pretrain_epoch=10)
        train_model(sess, model, analyze, train_data, valid_data)
        config.model_type = 'test'
        test_model(sess, model, analyze, test_data, vocab)

    if config.model_type == 'test' :
        print('establish the model...')
        # config.batch_size = len(test_data)
        model = Multi_Roles_Model.MultiRolesModel(config, vocab)
        analyze = Multi_Roles_Analyze.Multi_Roles_Analyze(config)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(sess, model, analyze, test_data, vocab)

    if config.model_type=='role_test':
        config.batch_size = 1
        config.model_type='test'
        print('establish the model...')
        model = Multi_Roles_Model.MultiRolesModel(config, vocab)
        analyze = Multi_Roles_Analyze.Multi_Roles_Analyze(config)
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_role_model(sess, model, analyze, test_data, vocab)

if __name__ == "__main__":
    tf.app.run()
