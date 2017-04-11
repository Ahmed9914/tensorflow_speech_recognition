#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import utils
import train_data

# Some configs
num_features = 13
epoch_save_step = 100 # save the checkpoint every

# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 2000
num_hidden = 200
num_layers = 1
batch_size = 16
initial_learning_rate = 1e-2
momentum = 0.9

# Training data

num_train_data = 10
num_test_data  = 5
wav_files_train, wav_files_test  = train_data.get_file_list(num_train_data, num_test_data, shuffle=False)

num_examples = len(wav_files_train)
num_batches_per_epoch = int(num_examples/batch_size)

train_inputs = train_data.prepare_inputs(wav_files_train)
train_targets = train_data.prepare_targets(wav_files_train)

test_inputs = train_data.prepare_inputs(wav_files_test)

# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    # IDK but adam gives better result
    #optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)
    optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))


def decode_single(session, test_input):

    val_feed = {
        inputs:  np.asarray([test_input]),
        seq_len: np.asarray([len(test_input)])
    }

    # Decoding
    d = session.run(decoded[0], feed_dict=val_feed)
    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

    seq = [s for s in dense_decoded[0] if s != -1]
    print('Decoded:\t%s' % (utils.decode_result(seq)))


with tf.Session(graph=graph) as session:

    saver = tf.train.Saver(tf.global_variables())
    snapshot = "ctc"
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
    last_epoch = 0

    if checkpoint:
        print("[i] LOADING checkpoint " + checkpoint)
        try:
            saver.restore(session, checkpoint)
            last_epoch = int(checkpoint.split('-')[-1]) + 1
            print("[i] start from epoch %d" % last_epoch)
        except:
            print("[!] incompatible checkpoint, restarting from 0")
    else:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()


    for curr_epoch in range(last_epoch, num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        try:
            for batch in range(num_batches_per_epoch):

                # Getting the index
                indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs = train_inputs[indexes]
                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = utils.pad_sequences(batch_train_inputs)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = utils.sparse_tuple_from(train_targets[indexes])

                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost*batch_size
                train_ler += session.run(ler, feed_dict=feed)*batch_size


            # Shuffle the data
            shuffled_indexes = np.random.permutation(num_examples)
            train_inputs = train_inputs[shuffled_indexes]
            train_targets = train_targets[shuffled_indexes]

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch, num_epochs, train_cost, train_ler, time.time() - start))

            if curr_epoch % epoch_save_step == 0 and curr_epoch > 0:
                print("[i] SAVING snapshot %s" % snapshot)
                saver.save(session, "checkpoints/" + snapshot + ".ckpt", curr_epoch)

        except KeyboardInterrupt:
            print("\nTest data:")
            for test in test_inputs:
                decode_single(session, test)

    print("FINISHED")
    print("Train data:")
    for test in train_inputs:
        decode_single(session, test)
    print("\nTest data:")
    for test in test_inputs:
        decode_single(session, test)
