import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import pdb
# SEED = 66478
linear = rnn_cell_impl._linear

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2]..
    """
    with tf.name_scope("add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


class MultiRolesModel():
    def __init__(self, config, vocab, my_embedding=None):
        self._vocab = vocab
        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._learn_rate = tf.Variable(float(config.learn_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = tf.assign(self._learn_rate, self._learn_rate * config.learning_rate_decay_factor)
        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learn_rate)
        self._opt=tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._interpose = config.interpose
        self._embedding_size = config.neurons
        self._layers = config.layers
        self._build_vars(config)
        self.model_type = config.model_type
        self._roles_number = config.roles_number
        self._max_grad_norm = config.max_grad_norm
        self._build_inputs()
        self.rl = config.rl
        self._my_embedding = my_embedding
        self.trained_embedding=config.trained_emb
        with tf.variable_scope('embedding'):
            if self.trained_embedding:
                self._word_embedding=tf.get_variable('embedding_word',shape=[self._vocab.vocab_size, config.neurons],
            			 initializer=tf.constant_initializer(value=self._my_embedding,dtype=tf.float32),trainable=True)
            else:
                self._word_embedding = tf.get_variable(name='embedding_word',
                                                   shape=[self._vocab.vocab_size, config.neurons])
            self._name_embedding = tf.get_variable(name='embedding_name',
                                                   shape=[self._roles_number + 1, config.neurons])
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

        def _encoding_roles(person_emb, name='',GPU_id=0):
            with tf.device('/device:GPU:%d' %GPU_id):
              with tf.variable_scope('encoding_role_' + name):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neurons, reuse=tf.get_variable_scope().reuse)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                encoding_cell = tf.contrib.rnn.DropoutWrapper(encoding_cell, 0.5, 1, 0.5)
                # for future test
                output, state_fw, state_bw = rnn.static_bidirectional_rnn(cell_fw=encoding_cell, cell_bw=encoding_cell,
                                                                          inputs=person_emb, dtype=tf.float32)
                # output, state_fw = rnn.static_rnn(encoding_cell, person_emb, dtype=tf.float32)

                state = tf.concat([state_fw, state_bw], -1)
                state = tf.matmul(state,tf.get_variable('Wi',[3,2 * self._embedding_size, self._embedding_size],
                                                          dtype=tf.float32,trainable=True))
                return output, state

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
            2)  # all_roles_state.shape=[layers,batch_size,roles_number,neurons] order by namelist

        with tf.variable_scope('rnn_encoding_context'):
          #with tf.device('/device:GPU:1'):
            # encoding_single_layer = tf.nn.rnn_cell.GRUCell(config.neurons)
            encoding_single_layer = tf.nn.rnn_cell.LSTMCell(config.neurons)
            encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
            context = tf.concat(values=[Chandler_emb, Joey_emb, Monica_emb, Phoebe_emb, Rachel_emb, Ross_emb, others_emb], axis=0)
            # context.shape=[(role_number*sentence_lens),batch_size,neurons]
            context = tf.unstack(context, axis=0) #context.shape=(role_number*sentence_lens) *[batch_size,neurons]
            # context_encoder, state_fw,state_bw = rnn.static_bidirectional_rnn(encoding_cell, encoding_cell, context,dtype=tf.float32)
            context_encoder, context_state_fw = rnn.static_rnn(encoding_cell, context, dtype=tf.float32)
          # context_encoder.shape same with context (role_number*sentence_lens) *[batch_size,neurons]
            self.context_vector = context_state_fw
            # top_output_context = [array_ops.reshape(o, [-1, 1, encoding_cell.output_size]) for o in context_encoder]
            # # top_output_context.shape=(role_number*sentence_lens)*[batch,1,neurons]
            # attention_states = array_ops.concat(top_output_context, 1)
            # #[batch,(role_number*sentence_lens),neurons]
            # attention_states_speaker = tf.split(attention_states, [-1, len(Monica_emb)], axis=1)[-1]
        # pdb.set_trace()
        with tf.variable_scope('cnn_encoding_context'):
            context=tf.stack([Chandler_emb, Joey_emb, Monica_emb, Phoebe_emb, Rachel_emb, Ross_emb, others_emb])
            context=tf.transpose(context,[2,1,3,0])
            # contex_filter=tf.Variable(tf.random_normal([3,self._embedding_size,7,3]))
            context_cnn=[]
            for filter_size in range(1,21):#[3,4,5]:
                context_filter = tf.Variable(tf.random_normal([filter_size, self._embedding_size, 7, 100]))
                context_bias=tf.get_variable("cnn_b_%s" % filter_size, shape=[100], initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
                context_conv=tf.nn.conv2d(context,context_filter,strides=[1,1,1,1],padding='VALID')
                cnn_h = tf.nn.relu(tf.nn.bias_add(context_conv, context_bias), name="relu")
                pooled = tf.nn.max_pool(cnn_h,ksize=[1, self._sentence_size - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
                # pooled=tf.squeeze(pooled)
                context_cnn.append(pooled)
            # pdb.set_trace()
            context_cnn_flat=tf.concat(context_cnn, 1)
            context_cnn_flat=tf.squeeze(context_cnn_flat)
            context_cnn_drop = tf.nn.dropout(context_cnn_flat, 0.5)
            attention_states_speaker= context_cnn_drop

        def _speaker_prediction(next_speaker_emb):
            with tf.variable_scope('speaker_prediction'):
                encoding_single_layer = tf.nn.rnn_cell.GRUCell(2 * config.neurons, reuse=tf.get_variable_scope().reuse)
                encoding_cell = tf.nn.rnn_cell.MultiRNNCell([encoding_single_layer] * config.layers)
                # pdb.set_trace()
                output = None
                state = encoding_cell.zero_state(self._batch_size, dtype=tf.float32)
                for emb in next_speaker_emb:
                    output, state = encoding_cell(emb, state=state)

            return output, state

        with tf.variable_scope('next_speaker'):
            next_speaker_emb = []
            for name_emb in name_list_emb:
                next_speaker_emb.append(tf.concat([name_emb, context_state_fw[-1][0]], 1))
            next_speaker_logit, _ = _speaker_prediction(next_speaker_emb)
            next_speaker_pred = linear(next_speaker_logit, self._roles_number,
                                       True)  # next_speaker.shape=[batch_size,roles_number]

            next_speaker = tf.nn.softmax(next_speaker_pred)  # next_speaker.shape=[batch_size,roles_number]
            self.next_speakers_vector = next_speaker

            next_speaker = tf.expand_dims(next_speaker, 0)  # next_speaker.shape=[1,batch_size,roles_number]
            next_speaker = tf.expand_dims(next_speaker, -1)  # next_speaker.shape=[1,batch_size,roles_number,1]

        def speaker_atten(encoder_state, attention_states, ans_emb, model_type='train'):
            with tf.variable_scope('speaker'):
                num_heads = 1
                batch_size = ans_emb[0].get_shape()[0]
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

                if model_type == 'test':
                    loop_function = extract_argmax_and_embed
                else:
                    loop_function = None

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
                    cell_de = tf.contrib.rnn.DropoutWrapper(cell_de, 0.5, 1, 0.5)
                    # cell_de = core_rnn_cell.OutputProjectionWrapper(cell_de, self._vocab_size)
                    outputs = []
                    prev = None
                    #   pdb.set_trace()
                    state = encoder_state
                    for i, inp in enumerate(ans_emb):
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

        def speaker_noatten(encoder_state, ans_emb, model_type='train'):
            with tf.variable_scope('speaker'):
                def extract_argmax_and_embed(prev, _):
                    """Loop_function that extracts the symbol from prev and embeds it."""
                    prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
                    return embedding_ops.embedding_lookup(self._word_embedding, prev_symbol)

                if model_type == 'train':
                    loop_function = None
                else:
                    loop_function = extract_argmax_and_embed

                linear = rnn_cell_impl._linear

                with tf.variable_scope("rnn_decoder"):
                    single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    cell_de = tf.nn.rnn_cell.MultiRNNCell([single_cell_de] * self._layers)
                    # cell_de = core_rnn_cell.OutputProjectionWrapper(cell_de, self._vocab_size)
                    outputs = []
                    prev = None
                    #   pdb.set_trace()
                    state = encoder_state
                    for i, inp in enumerate(ans_emb):
                        if loop_function is not None and prev is not None:
                            with tf.variable_scope("loop_function", reuse=True):
                                inp = array_ops.stop_gradient(loop_function(prev, i))
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        output, state = cell_de(inp, state)
                        #  pdb.set_trace()
                        with tf.variable_scope('OutputProjecton'):
                            output = linear([output], self._vocab.vocab_size, True)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = array_ops.stop_gradient(output)

                outputs = tf.transpose(outputs, perm=[1, 0, 2])
                return outputs

        def speaker_beam(embedding_word, encoder_state, ans_emb, beam_size=10, model_type='train',
                         output_projection=None):
            with tf.variable_scope('speaker'):
             # with tf.device('/device:GPU:1'):
                # pdb.set_trace()
                num_symbols = embedding_word.get_shape()[0].value
                embedding_size = embedding_word.get_shape()[1].value

                def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols):
                    if output_projection is not None:
                        prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])

                    probs = tf.log(tf.nn.softmax(prev))
                    if i == 1:
                        probs = tf.reshape(probs[0, :], [-1, num_symbols])
                    if i > 1:
                        probs = tf.reshape(probs + log_beam_probs[-1], [-1, beam_size * num_symbols])

                    best_probs, indices = tf.nn.top_k(probs, beam_size)
                    indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
                    best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

                    symbols = indices % num_symbols  # Which word in vocabulary.
                    beam_parent = indices // num_symbols  # Which hypothesis it came from.
                    beam_symbols.append(symbols)
                    beam_path.append(beam_parent)
                    log_beam_probs.append(best_probs)

                    emb_prev = embedding_ops.embedding_lookup(embedding_word, symbols)
                    emb_prev = tf.reshape(emb_prev, [-1, embedding_size])
                    return emb_prev

                if model_type == 'train':
                    loop_function = None
                else:
                    loop_function = loop_function

                linear = rnn_cell_impl._linear

                with tf.variable_scope("rnn_decoder"):
                    single_cell_de = tf.nn.rnn_cell.GRUCell(self._embedding_size)
                    cell_de = tf.nn.rnn_cell.MultiRNNCell([single_cell_de] * self._layers)
                    # cell_de = core_rnn_cell.OutputProjectionWrapper(cell_de, self._vocab_size)
                    outputs = []
                    prev = None
                    #   pdb.set_trace()
                    state = encoder_state
                    for i, inp in enumerate(ans_emb):
                        log_beam_probs, beam_path, beam_symbols = [], [], []
                        if loop_function is not None and prev is not None:
                            with tf.variable_scope("loop_function", reuse=True):
                                inp = array_ops.stop_gradient(
                                    loop_function(prev, i, log_beam_probs, beam_path, beam_symbols))
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        output, state = cell_de(inp, state)
                        #  pdb.set_trace()
                        with tf.variable_scope('OutputProjecton'):
                            output = linear([output], self._vocab.vocab_size, True)
                        outputs.append(output)
                        if loop_function is not None:
                            prev = array_ops.stop_gradient(output)

                outputs = tf.transpose(outputs, perm=[1, 0, 2])
                return outputs

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


            state_all_roles_speaker_ = tf.multiply(state_all_roles,
                                                  next_speaker)  # state_all_roles_speaker.shape=[layers,batch_size,roles_number,neurons]
            state_all_roles_speaker_matrix = tf.reduce_sum(state_all_roles_speaker_, 2)
            state_all_roles_speaker = tf.unstack(state_all_roles_speaker_matrix)
            # next_speakers=tf.argmax(next_speaker,1)

            # cnn all state of context to concat with answer_emb,but experiment is bad
            # state_concate=tf.transpose(state_all_roles,[1,2,3,0])
            # state_concate=tf.reshape(state_concate,[config.batch_size,config.roles_number,-1])
            # state_concate=tf.expand_dims(state_concate,-1)
            # state_concate=tf.tile(state_concate,[1,1,1,config.sentence_size])
            # state_concate = tf.transpose(state_concate,[0,3,2,1])
            # answer_concate=tf.stack(answer_emb)
            # answer_concate=tf.expand_dims(answer_concate,-1)
            # answer_concate=tf.tile(answer_concate,[1,1,1,config.roles_number])
            # answer_concate=tf.transpose(answer_concate,[1,0,2,3])
            # state_answer_concate=tf.concat([state_concate,answer_concate],2)
            #
            #
            # state_filter = tf.Variable(tf.random_normal([2,self._embedding_size, 7,1]))
            # state_conv = tf.nn.conv2d(state_answer_concate, state_filter, strides=[1, 1, 1, 1], padding='SAME')
            # state_pool=tf.nn.max_pool(state_conv,[1,2,2,1],[1,1,4,1],'SAME')
            #
            # state_emb=tf.squeeze(state_pool)
            # state_emb=tf.transpose(state_emb,[1,0,2])
            # state_emb=tf.unstack(state_emb,axis=0)
            # answer_emb=state_emb

            # state_emb=tf.nn.relu(tf.matmul(state_pool, fc1_weights) + fc1_biases)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.

            if config.beam:
                response = speaker_beam(self._word_embedding, state_all_roles_speaker, answer_emb,
                                                model_type=self.model_type)
            else:
                if config.attention:
                    response = speaker_atten(state_all_roles_speaker, attention_states_speaker, answer_emb,
                                             self.model_type)
                else:
                    response = speaker_noatten(state_all_roles_speaker, answer_emb, self.model_type)


        with tf.variable_scope('loss_function'):
         # with tf.device('/device:GPU:1'):
            # Our targets are decoder inputs shifted by one.
            # targets = [self.decoder_inputs[i + 1]
            #           for i in xrange(len(self.decoder_inputs) - 1)]
            _, labels = tf.split(self._answers, [1, -1], 1)
            labels = tf.concat([labels, _], axis=1)
            true_speaker = self._speaker

            cross_entropy_speaker = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=next_speaker_pred,
                                                                                   labels=true_speaker,
                                                                                   name='cross_entrooy_speaker')
            cross_entropy_sentence = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=response,
                                                                                    labels=labels,
                                                                                    name="cross_entropy_sents")
            cross_entropy_speaker = tf.reduce_mean(cross_entropy_speaker,name="cross_entropy_speaker")

            cross_entropy_sentence = tf.multiply(cross_entropy_sentence, self._weight)  # batch_size * sents_size

            cross_entropy_sentence = tf.reduce_sum(cross_entropy_sentence, axis=1)
            weight_sum = tf.reduce_sum(self._weight, axis=1)
            cross_entropy_sentence = cross_entropy_sentence / weight_sum

            if self.rl:
                cross_entropy_sentence = cross_entropy_sentence * self.rl_reward
            cross_entropy_sentence_sum = tf.reduce_mean(cross_entropy_sentence, name="cross_entropy_sentences")

            self.loss = cross_entropy_sentence_sum  # + cross_entropy_speaker
            # self.loss = 0.4*cross_entropy_sentence_sum + 0.6*cross_entropy_speaker

        grads_and_vars = []
            # grads_and_vars=self._opt.compute_gradients(self.loss)
        grads_and_vars.append(self._opt.compute_gradients(cross_entropy_sentence_sum))
        pdb.set_trace()
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
            if len(grads) == 0: pdb.set_trace() # some variable get none grads
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

    def _build_vars(self,config):

        self.rl_reward = tf.get_variable('rl_reward', [1], dtype=tf.float32, trainable=False)
        # fc1_weights = tf.Variable(  # fully connected.
        #     tf.truncated_normal([50,config.neurons],
        #                         stddev=0.1,
        #                         seed=SEED
        #                         ))
        # fc1_biases = tf.Variable(tf.constant(0.1, shape=[config.neurons], dtype=tf.float32))
        # init=tf.random_normal_initializer(stddev=0.1)
        # self._w_context= tf.Variable(tf.truncated_normal([config.batch_size,1,1,config.sentence_size],stddev=0.1,seed=SEED ))
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
                answer.append(i.get('answer'))
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
            # try:
            loss, _, summary = sess.run(output_list, feed_dict=feed_dict)
            # except:
            #     pdb.set_trace()

            return loss, _, summary
        if step_type == 'test':
            output_list = [self.loss, self.response, self.merged, self.next_speakers_vector]
            try:
                loss, response, summary, next_speakers_vector = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return loss, response, summary, next_speakers_vector
        if step_type == 'rl_compute':
            output_list = [self.context_vector]
            try:
                context_vector = sess.run(output_list, feed_dict=feed_dict)
            except:
                pdb.set_trace()
            return context_vector
        if step_type == 'rl_learn':
            # pdb.set_trace()
            self.rl_reward = data_dict['reward']
            output_list = [self.loss, self.train_op, self.merged]
            loss, _, _ = sess.run(output_list, feed_dict=feed_dict)
            return loss

        print('step_type is wrong!>>>')
        return None
