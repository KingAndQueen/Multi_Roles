import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import pdb


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2]..
    """
    with tf.name_scope("add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MultiRolesModel():
    def __init__(self, config, vocab):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = tf.assign(self._learn_rate, self._learn_rate * config.learning_rate_decay_factor)
        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        self._interpose = config.interpose
        self._embedding_size = config.neurons
        self._layers = config.layers
        self._build_vars()
        self.model_type = config.model_type
        self._roles_number = config.roles_number
        self._max_grad_norm = config.max_grad_norm
        self._build_inputs()
        self.rl = config.rl
        with tf.variable_scope('embedding'):
            self._word_embedding = tf.get_variable(name='embedding_word',
                                                   shape=[self._vocab.vocab_size, config.neurons])
            self._name_embedding=tf.get_variable(name='emnbedding_name',shape=[self._roles_number+1,config.neurons])
            _Monica = tf.unstack(self._Monica, axis=1)
            Monica_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Monica]
            _Joey = tf.unstack(self._Joey, axis=1)
            Joey_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Joey]
            _Chandler = tf.unstack(self._Chandler, axis=1)
            Chandler_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Chandler]
            _Phoebe = tf.unstack(self._Phoebe, axis=1)
            Phoebe_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Phoebe]
            _Rachel = tf.unstack(self._Rachel, axis=1)
            Rachel_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Rachel]
            _Ross = tf.unstack(self._Ross, axis=1)
            Ross_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _Ross]
            _others = tf.unstack(self._others, axis=1)
            others_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _others]
            _answer = tf.unstack(self._answers, axis=1)
            answer_emb = [tf.nn.embedding_lookup(self._word_embedding, word) for word in _answer]
            _name_list = tf.unstack(self._name_list, axis=1)
            # word_embedding may be too big for name
            name_list_emb = [tf.nn.embedding_lookup(self._name_embedding, word) for word in _name_list]

        def _encoding_roles(person_emb, name=''):
            with tf.variable_scope('encoding_role_' + name):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neurons, reuse=tf.get_variable_scope().reuse)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                # for future test
                # output, state_fw,state_bw = rnn.static_bidirectional_rnn(cell_fw=encoding_cell, cell_bw=encoding_cell,
                # inputs=person_emb, dtype=tf.float32)
                output, state_fw = rnn.static_rnn(encoding_cell, person_emb, dtype=tf.float32)
                return output, state_fw

        # encoder different roles
        # encoding_roles_functions={
        #     'Monica':  _encoding_roles(Monica_emb,'monica')
        #
        # }
        monica_encoder, monica_state = _encoding_roles(Monica_emb,
                                                       'Monica')  # monica_sate.shape=layers*[batch_size,neurons]
        joey_encoder, joey_state = _encoding_roles(Joey_emb, 'Joey')
        chandler_encoder, chandler_state = _encoding_roles(Chandler_emb, 'Chandler')
        phoebe_encoder, phoebe_state = _encoding_roles(Phoebe_emb, 'Phoebe')
        rachel_encoder, rachel_state = _encoding_roles(Rachel_emb, 'Rachel')
        ross_encoder, ross_state = _encoding_roles(Ross_emb, 'Ross')
        others_encoder, others_state = _encoding_roles(others_emb, 'others')

        monica_state = tf.expand_dims(tf.stack(monica_state), 2)  # monica_sate.shape=[layers,batch_size,1,neurons]
        joey_state = tf.expand_dims(tf.stack(joey_state), 2)
        chandler_state = tf.expand_dims(tf.stack(chandler_state), 2)
        phoebe_state = tf.expand_dims(tf.stack(phoebe_state), 2)
        rachel_state = tf.expand_dims(tf.stack(rachel_state), 2)
        ross_state = tf.expand_dims(tf.stack(ross_state), 2)
        others_state = tf.expand_dims(tf.stack(others_state), 2)
        state_all_roles = tf.concat(
            [chandler_state, joey_state, monica_state, phoebe_state, rachel_state, ross_state, others_state],
            2)  # all_roles_sate.shape=[layers,batch_size,roles_number,neurons] order by namelist

        with tf.variable_scope('encoding_context'):
            encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neurons)
            encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
            context = tf.concat(values=[Chandler_emb, Joey_emb, Monica_emb, Phoebe_emb, Rachel_emb, Ross_emb,others_emb], axis=0)
            context = tf.unstack(context, axis=0)
            # context_encoder, state_fw,state_bw = rnn.static_bidirectional_rnn(encoding_cell, encoding_cell, context,dtype=tf.float32)
            context_encoder, context_state_fw = rnn.static_rnn(encoding_cell, context, dtype=tf.float32)
            self.context_vector=context_state_fw
            top_output_context = [array_ops.reshape(o, [-1, 1, encoding_cell.output_size]) for o in context_encoder]
            attention_states = array_ops.concat(top_output_context, 1)
        linear = rnn_cell_impl._linear

        def _speaker_prediction(next_speaker_emb):
            with tf.variable_scope('speaker_prediction'):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(2*config.neurons, reuse=tf.get_variable_scope().reuse)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                # pdb.set_trace()
                output=None
                state=encoding_cell.zero_state(self._batch_size, dtype=tf.float32)
                for emb in next_speaker_emb:
                    output, state = encoding_cell(emb, state=state)

            return output, state

        with tf.variable_scope('next_speaker'):
            next_speaker_emb=[]
            for name_emb in name_list_emb:
                next_speaker_emb.append(tf.concat([name_emb, context_state_fw[-1]], 1))
            next_speaker_logit, _ = _speaker_prediction(next_speaker_emb)
            next_speaker_pred = linear(next_speaker_logit, self._roles_number,
                                       True)  # next_speaker.shape=[batch_size,roles_number]

            next_speaker = tf.nn.softmax(next_speaker_pred) # next_speaker.shape=[batch_size,roles_number]
            self.next_speakers_vector = next_speaker
            next_speaker = tf.expand_dims(next_speaker, 0)  # next_speaker.shape=[1,batch_size,roles_number]
            next_speaker = tf.expand_dims(next_speaker, -1)  # next_speaker.shape=[1,batch_size,roles_number,1]

        def speaker(encoder_state, attention_states, q_emb, model_type='train'):
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

                if model_type == 'train':
                    loop_function = None
                if model_type == 'test':
                    loop_function = extract_argmax_and_embed

                linear = rnn_cell_impl._linear
                batch_attn_size = array_ops.stack([batch_size, attn_size])
                attns = [array_ops.zeros(batch_attn_size, dtype=tf.float32) for _ in range(num_heads)]
                for a in attns:
                    a.set_shape([None, attn_size])

                with tf.variable_scope("rnn_decoder"):
                    single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    # $ cell_de = single_cell_de
                    # if self._layers > 1:
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
            # first decide wheter to speake,then choose the speaker
            # gate_decision_lastSpeaker=tf.tanh(tf.matmul(self._w_context,tf.matmul(monica_encoder[-1],tf.transpose(context_encoder[-1]))))
            # gate_GRU = tf.nn.rnn_cell.GRUCell(1)
            # temp1=tf.squeeze(top_output_context[-1])
            # temp2=tf.transpose(monica_encoder[-1])
            # gate_input=tf.matmul(self._w_attention,tf.matmul(temp1,temp2))
            # #gate_input=tf.unstack(gate_input,axis=1)
            # gate_input=tf.transpose(gate_input)
            # decision_contest, state_gate = rnn.static_rnn(cell=gate_GRU,inputs=[gate_input],dtype=tf.float32)
            # decision=tf.sigmoid(tf.add(tf.squeeze(decision_contest),tf.squeeze(gate_decision_lastSpeaker)))
            #    if decision > self._interpose:

            attention_states_speaker = tf.split(attention_states, [-1, len(Monica_emb)], axis=1)[-1]

            state_all_roles_speaker = tf.multiply(state_all_roles,
                                                  next_speaker)  # state_all_roles_speaker.shape=[layers,batch_size,roles_number,neurons]
            state_all_roles_speaker = tf.reduce_sum(state_all_roles_speaker, 2)
            state_all_roles_speaker = tf.unstack(state_all_roles_speaker)
            # next_speakers=tf.argmax(next_speaker,1)

            response = speaker(state_all_roles_speaker, attention_states_speaker, answer_emb, self.model_type)
            #     else:
            #         response=[]

        with tf.variable_scope('loss_function'):
            # Our targets are decoder inputs shifted by one.
            # targets = [self.decoder_inputs[i + 1]
            #           for i in xrange(len(self.decoder_inputs) - 1)]
            _, labels = tf.split(self._answers, [1, -1], 1)
            labels = tf.concat([labels, _], axis=1)
            true_speaker = self._speaker


            cross_entropy_speaker = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=next_speaker_pred,
                                                                                   labels=true_speaker)
            cross_entropy_sentence = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response,
                                                                                    labels=labels,
                                                                                    name="cross_entropy")
            cross_entropy_speaker = tf.reduce_sum(cross_entropy_speaker)

            cross_entropy_sentence = tf.multiply(cross_entropy_sentence, self._weight)
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            cross_entropy_sentence = tf.reduce_sum(cross_entropy_sentence, axis=1)
            cross_entropy_sentence = cross_entropy_sentence / weight_sum

            if self.rl:
                cross_entropy_sentence = cross_entropy_sentence * self.rl_reward
            cross_entropy_sentence_sum = tf.reduce_sum(cross_entropy_sentence, name="cross_entropy_sum")

            self.loss = cross_entropy_sentence_sum + cross_entropy_speaker
            # self.loss = 0.4*cross_entropy_sentence_sum + 0.6*cross_entropy_speaker
        grads_and_vars = []
        #grads_and_vars.append(self._opt.compute_gradients(self.loss))
        grads_and_vars.append(self._opt.compute_gradients(cross_entropy_sentence_sum))
        grads_and_vars.append(self._opt.compute_gradients(cross_entropy_speaker))
        grads_and_vars = self._combine_gradients(grads_and_vars)

        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]

        self.train_op = self._opt.apply_gradients(grads_and_vars=grads_and_vars, name='train_op')

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        self.response = tf.argmax(response, axis=2)
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.learning_rate_summary = tf.summary.scalar("learning_rate",
                                                       self._learn_rate)
        self.merged = tf.summary.merge_all()

    def _combine_gradients(self, tower_grads):
        combine_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                if g is None:
                    continue
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            if len(grads) == 0: pdb.set_trace()
            grad = tf.concat(grads, 0)
            grad = tf.reduce_sum(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            combine_grads.append(grad_and_var)
        return combine_grads

    def _build_inputs(self):
        self._Monica = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Monica')
        self._Joey = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Joey')
        self._Chandler = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Chandler')
        self._Phoebe = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Phoebe')
        self._Rachel = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Rachel')
        self._Ross = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='Ross')
        self._others = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='others')
        self._weight = tf.placeholder(tf.float32, [self._batch_size, self._sentence_size], name='weight')
        self._answers = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name='answer')
        self._name_list = tf.placeholder(tf.int32, [self._batch_size, self._roles_number], name='name_list')
        self._speaker = tf.placeholder(tf.int32, [self._batch_size], name='true_speaker')

    def _build_vars(self):

        self.rl_reward = tf.get_variable('rl_reward', [1], dtype=tf.float32, trainable=False)
        # init=tf.random_normal_initializer(stddev=0.1)
        # self._w_context=init([1,self._batch_size])
        # self._w_attention=init([1,self._batch_size])
        #   self._w_transt=init([self._embedding_size,2*self._embedding_size])

    def get_batch(self, data_raw):
        # return a list of batches
        list_all_batch = []

        #    pdb.set_trace()
        for _ in range(0, len(data_raw), self._batch_size):
            if _ + self._batch_size > len(data_raw): continue
            data_batch = data_raw[_:_ + self._batch_size]
            Monica, Joey, Chandler, Phoebe, Rachel, Ross, others, answer, weight, name_list, speaker = [], [], [], [], [], [], [], [], [], [], []
            for i in data_batch:
                if 'Monica' in i:
                    Monica.append(i.get('Monica'))
                else:
                    Monica.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
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
                if 'Ross' in i:
                    Ross.append(i.get('Ross'))
                else:
                    Ross.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                if 'others' in i:
                    others.append(i.get('others'))
                else:
                    others.append(self._sentence_size * [self._vocab.word_to_index('<pad>')])
                name_list.append(i.get('name_list'))
                answer.append(i.get('ans'))
                weight.append(i.get('weight'))
                speaker.append(i.get('speaker'))

            list_all_batch.append({'Monica': Monica, 'Joey': Joey, 'Chandler': Chandler, 'Phoebe': Phoebe,
                                   'Rachel': Rachel, 'Ross': Ross, 'others': others, 'answer': answer, 'weight': weight,
                                   'name_list': name_list, 'speaker': speaker})
            # pdb.set_trace()
        return list_all_batch

    def step(self, sess, data_dict, step_type='train'):
        self.model_type = step_type
        feed_dict = {self._Ross: data_dict['Ross'],
                     self._Rachel: data_dict['Rachel'],
                     self._Phoebe: data_dict['Phoebe'],
                     self._Chandler: data_dict['Chandler'],
                     self._Monica: data_dict['Monica'],
                     self._Joey: data_dict['Joey'],
                     self._others: data_dict['others'],
                     self._answers: data_dict['answer'],
                     self._weight: data_dict['weight'],
                     self._name_list: data_dict['name_list'],
                     self._speaker: data_dict['speaker']}
        if step_type == 'train':
            output_list = [self.loss, self.train_op, self.merged]
            try:
                loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()

            return loss, _, summary
        if step_type == 'test':
            output_list = [self.loss, self.response, self.merged, self.next_speakers_vector]
            loss, response, summary, next_speakers_vector = sess.run(output_list, feed_dict=feed_dict)
            return loss, response, summary, next_speakers_vector
        if step_type == 'rl_compute':
            output_list = [ self.context_vector]
            context_vector=sess.run(output_list, feed_dict=feed_dict)
            return context_vector
        if step_type == 'rl_learn':
            self.rl_reward = data_dict['reward']
            output_list = [self.loss, self.train_op, self.merged]
            loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            return loss, _, summary
        print('step_type is wrong!>>>')
        return None
        # try:
        #     loss,_=sess.run(output_list, feed_dict=feed_dict)
        # except:
        #     pdb.set_trace()
