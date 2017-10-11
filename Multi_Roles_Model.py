import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import embedding_ops
from tensorflow.python.util import nest
import pdb

class MuliRolesModel():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._opt = tf.train.GradientDescentOptimizer(learning_rate=config.learn_rate)
        self._build_inputs()
        self._interpose = config.interpose
        self._embedding_size=config.neuros
        self._layers=config.layers
        self._build_vars()
        self._build_inputs()
        with tf.variable_scope('embedding'):
            self._word_embedding = tf.get_variable('embedding_word', [self._vocab.vocab_size, config.neuros])
            _Monica = tf.unstack(self._Monica, axis=1)
            Monica_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Monica]
            _Joey = tf.unstack(self._Joey, axis=1)
            Joey_emb = [tf.nn.embedding_lookup(self._word_embedding,word) for word in _Joey]
            _Chandler= tf.unstack(self._Chandler, axis=1)
            Chandler_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Chandler]
            _Phoebe=tf.unstack(self._Phoebe, axis=1)
            Phoebe_emb = [tf.nn.embedding_lookup(self._word_embedding,word)for word in _Phoebe]
            _Rachel=tf.unstack(self._Rachel, axis=1)
            Rachel_emb = [tf.nn.embedding_lookup(self._word_embedding,word) for word in _Rachel]
            _Ross =tf.unstack(self._Ross, axis=1)
            Ross_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Ross]

        def _encoding_roles( person_emb):
            with tf.variable_scope('encoding_role'):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neuros)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                # for future test
                #output, state_fw,state_bw = rnn.static_bidirectional_rnn(cell_fw=encoding_cell, cell_bw=encoding_cell,
                                                                                    # inputs=person_emb, dtype=tf.float32)
                output, state_fw = rnn.static_rnn(encoding_cell, person_emb, dtype=tf.float32)
                return output, state_fw

        # encoder different roles
        monica_encoder, monica_state = _encoding_roles(Monica_emb)

        with tf.variable_scope('encoding_context'):
            encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neuros)
            encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
            context = tf.concat(values=[Monica_emb, Joey_emb, Chandler_emb, Phoebe_emb, Rachel_emb, Ross_emb], axis=0)
            context=tf.unstack(context,axis=0)
           # context_encoder, state_fw,state_bw = rnn.static_bidirectional_rnn(encoding_cell, encoding_cell, context,dtype=tf.float32)
            context_encoder, context_state_fw= rnn.static_rnn(encoding_cell, context,dtype=tf.float32)
            top_output_context = [array_ops.reshape(o, [-1, 1, encoding_cell.output_size]) for o in context_encoder]
            attention_states = array_ops.concat(top_output_context, 1)


        def speaker( encoder_state, attention_states, q_emb, logits_MemKG=None):
            with tf.variable_scope('speaker'):
                num_heads = 1
                batch_size = q_emb[0].get_shape()[0]
                attn_length = attention_states.get_shape()[1].value
                attn_size = attention_states.get_shape()[2].value
                hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
                hidden_features = []
                v = []
                attention_vec_size = attn_size
                for a in range(num_heads):
                    k = tf.get_variable('AttnW_%d' % a, [1, 1, attn_size, attention_vec_size])
                    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], 'SAME'))
                    v.append(tf.get_variable('AttnV_%d' % a, [attention_vec_size]))

                def attention(query):
                    ds = []
                    if nest.is_sequence(query):
                        query_list = nest.flatten(query)
                        for q in query_list:
                            ndims = q.get_shape().ndims
                            if ndims:
                                assert ndims == 2
                        query = array_ops.concat(query_list, 1)
                    for a in range(num_heads):
                        with tf.variable_scope('Attention_%d' % a):
                            y = linear(query, attention_vec_size, True)
                            y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                            s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                            a = nn_ops.softmax(s)
                            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                            ds.append(array_ops.reshape(d, [-1, attn_size]))
                    return ds

                def extract_argmax_and_embed(prev, _):
                    """Loop_function that extracts the symbol from prev and embeds it."""
                    prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
                    return embedding_ops.embedding_lookup(self._word_embedding, prev_symbol)


                loop_function = extract_argmax_and_embed
                linear = rnn_cell_impl._linear
                batch_attn_size = array_ops.stack([batch_size, attn_size])
                attns = [array_ops.zeros(batch_attn_size, dtype=tf.float32) for _ in range(num_heads)]
                for a in attns:
                    a.set_shape([None, attn_size])

                with tf.variable_scope("rnn_decoder"):
                    single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    cell_de = single_cell_de
                    if self._layers > 1:
                        cell_de = tf.nn.rnn_cell.MultiRNNCell([single_cell_de] * self._layers)
                        # cell_de = core_rnn_cell.OutputProjectionWrapper(cell_de, self._vocab_size)
                    outputs = []
                    prev = None
                    #   pdb.set_trace()
                    state = encoder_state
                    for i, inp in enumerate(q_emb):
                        if loop_function is not None and prev is not None:
                            with tf.variable_scope("loop_function", reuse=True):
                                inp = array_ops.stop_gradient(loop_function(prev, i))

                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        inp = linear([inp] + attns, self._embedding_size, True)

                        output, state = cell_de(inp, state)
                        attns = attention(state)
                        #  pdb.set_trace()
                        with tf.variable_scope('AttnOutputProjecton'):
                            output = linear([output] + attns, self._vocab.vocab_size, True)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = array_ops.stop_gradient(output)

                outputs = tf.transpose(outputs, perm=[1, 0, 2])
                return outputs  # ,outputs_original

        with tf.variable_scope('interaction'):
            gate_decision_lastSpeaker=tf.tanh(tf.matmul(self._w_context,tf.matmul(monica_encoder[-1],tf.transpose(context_encoder[-1]))))
            gate_GRU = tf.nn.rnn_cell.GRUCell(1)
            temp1=tf.squeeze(top_output_context[-1])
            temp2=tf.transpose(monica_encoder[-1])
            gate_input=tf.matmul(self._w_attention,tf.matmul(temp1,temp2))
            #gate_input=tf.unstack(gate_input,axis=1)
            gate_input=tf.transpose(gate_input)
            decision_contest, state_gate = rnn.static_rnn(cell=gate_GRU,inputs=[gate_input],dtype=tf.float32)
            decision=tf.sigmoid(tf.add(tf.squeeze(decision_contest),tf.squeeze(gate_decision_lastSpeaker)))
        #    if decision > self._interpose:

            attention_states_speaker=tf.split(attention_states,[-1,len(Monica_emb)],axis=1)[-1]
            response=speaker(context_state_fw,attention_states_speaker,Monica_emb)
       #     else:
       #         response=[]


        with tf.variable_scope('loss_function'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response,
                                                                        labels = self._answers,
                                                                        name = "cross_entropy")
            cross_entropy = cross_entropy * self._weight
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
            cross_entropy = cross_entropy / weight_sum
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
            self.loss=cross_entropy_sum
        grads_and_vars = self._opt.compute_gradients(cross_entropy_sum)
        self.train_op = self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')

        self.saver = tf.train.Saver(tf.global_variables())

        self.response = tf.argmax(response, axis=2)
    def _build_inputs(self):
        self._Monica = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Monica')
        self._Joey = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Joey')
        self._Chandler = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Chandler')
        self._Phoebe = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Phoebe')
        self._Rachel = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Rachel')
        self._Ross = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Ross')
        self._weight=tf.placeholder(tf.float32,[self._batch_size,self._sentence_size],name='weight')
        self._answers=tf.placeholder(tf.int32,[self._batch_size,self._sentence_size],name='answer')
    def _build_vars(self):
        init=tf.random_normal_initializer(stddev=0.1)
        self._w_context=init([1,self._batch_size])
        self._w_attention=init([1,self._batch_size])
     #   self._w_transt=init([self._embedding_size,2*self._embedding_size])

    def get_batch(self,data_raw):
        # return a list of batches
        list_all_batch=[]

    #    pdb.set_trace()
        for _ in range(0,len(data_raw),self._batch_size):
            if _+self._batch_size>len(data_raw):continue
            data_batch=data_raw[_:_+self._batch_size]
            Monica,Joey,Chandler,Phoebe,Rachel,Rose,answer,weight=[],[],[],[],[],[],[],[]
            for i in data_batch:
                if 'Monica' in i:
                    Monica.append(i.get('Monica'))
                else:
                    Monica.append(self._sentence_size*[self._vocab.word_to_index('<pad>')])
                if 'Joey' in i:
                    Joey.append(i.get('Joey'))
                else:
                    Joey.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                if 'Chandler' in i:
                    Chandler.append(i.get('Chandler'))
                else:
                    Chandler.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                if 'Phoebe' in i:
                    Phoebe.append(i.get('Phoebe'))
                else:
                    Phoebe.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                if 'Rachel' in i:
                    Rachel.append(i.get('Rachel'))
                else:
                    Rachel.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                if 'Rose' in i:
                    Rose.append(i.get('Rose'))
                else:
                    Rose.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])

                answer.append(i.get('ans'))
                weight.append(i.get('weight'))

            list_all_batch.append({'Monica':Monica,'Joey':Joey,'Chandler':Chandler,'Phoebe':Phoebe,
                 'Rachel':Rachel,'Rose':Rose,'answer':answer,'weight':weight})
       # pdb.set_trace()
        return  list_all_batch


    def step(self, sess, data_dict,step_type='train'):

        feed_dict = {self._Ross:data_dict['Rose'],self._Rachel:data_dict['Rachel'],self._Phoebe:data_dict['Phoebe'],
                     self._Chandler:data_dict['Chandler'],self._Monica:data_dict['Monica'],self._Joey:data_dict['Joey'],
                     self._answers:data_dict['answer'],self._weight:data_dict['weight']}
        if step_type=='train':
            output_list=[self.loss,self.train_op]
        else:
            output_list=[self.loss,self.response]
        loss,_=sess.run(output_list, feed_dict=feed_dict)
        return loss, _
