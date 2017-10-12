import tensorflow as tf
import numpy as np
import Multi_Roles_Model
import Multi_Roles_Data
import os
from sklearn import model_selection
#set parameters of model
flags=tf.app.flags
flags.DEFINE_string('model_type','test','whether model initial from checkpoints')
flags.DEFINE_string('data_dir','data/','data path for model')
flags.DEFINE_string('checkpoints_dir','checkpoints/','path for save checkpoints')
flags.DEFINE_string('device_type','gpu','device for computing')

flags.DEFINE_integer('layers',3,'levels of rnn or cnn')
flags.DEFINE_integer('neuros',128,'neuros number of one level')
flags.DEFINE_integer('batch_size',128, 'batch_size')
flags.DEFINE_integer('roles_number',6,'number of roles in the data')
flags.DEFINE_integer('epoch',500,'training times' )
flags.DEFINE_integer('check_epoch',50,'training times' )
flags.DEFINE_integer('sentence_size',20,'length of sentence')
flags.DEFINE_float('interpose',0.5,'value for gru gate to decide interpose')
flags.DEFINE_float('learn_rate',0.001,'value for gru gate to decide interpose')

config=flags.FLAGS


def show_result(seq, vocab):
    words = []
    if isinstance(seq, (list, np.ndarray)):
        for idx in seq:
            if isinstance(idx,(list,np.ndarray)):
                show_result(idx,vocab)
            else:
                words.append(vocab.index_to_word(idx))
        print (words)
    if isinstance(seq, (str, int)):
        print (vocab.idx_to_word(seq))

def data_process(config):
#read data from file and normalized
    vocabulary=Multi_Roles_Data.Vocab()
    train_data,test_data=Multi_Roles_Data.get_data(config.data_dir,vocabulary,config.sentence_size)
    print('vocab size:',vocabulary.vocab_size)
    return train_data,test_data,vocabulary

#training model
def train_model(sess,model,train_data):
    train_data,eval_data=model_selection.train_test_split(train_data,test_size=0.2)

    current_step=1
    data_input_train = model.get_batch(train_data)
    data_input_eval=model.get_batch(eval_data)
    while current_step<config.epoch:
      #  print ('current_step:',current_step)
        total_loss = 0.0
        for i in range(len(data_input_train)):
            loss,_=model.step(sess,data_input_train[i])
            total_loss+=loss
        if current_step%config.check_epoch==0:
            print ('current_step:', current_step)
            print ('training total loss:',total_loss)
            loss_eval=0.0
            for i in range(len(data_input_eval)):
                loss,_=model.step(sess,data_input_eval[i])
                loss_eval+=loss
            print ('evaluation total loss:',loss_eval)

        current_step+=1
    checkpoint_path=os.path.join(config.checkpoints_dir,'MultiRoles.ckpt')
    model.saver.save(sess,checkpoint_path,global_step=current_step)

def test_model(sess,model,test_data,vocab):
    data_input_test=model.get_batch(test_data)
    loss_test=0.0
    predicts=[]
    for data_test in data_input_test:
        loss,predict=model.step(sess,data_test,step_type='test')
        loss_test+=loss
        predicts.append(predict)
        show_result(predict, vocab)
    print('test total loss:',loss_test)

#testing model
def main(_):
    train_data,test_data,vocab=data_process(config)
    # initiall model from new parameters or checkpoints
    sess = tf.Session()
    model = Multi_Roles_Model.MuliRolesModel(config,vocab)
    if config.model_type == 'training' or config.model_type == 'train' :
        print('Initial model with fresh parameters....')
        sess.run(tf.global_variables_initializer())
        train_model(sess, model,train_data)
    if config.model_type == 'testing' or config.model_type == 'test':
        print('Reload model from checkpoints.....')
        ckpt = tf.train.get_checkpoint_state(config.checkpoints_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        test_model(sess,model,test_data,vocab)


if __name__ == "__main__":
    tf.app.run()