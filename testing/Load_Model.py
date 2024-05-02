import math
import os
import sys

import Read_Data as rd
import tensorflow as tf
#tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from script_parameters import sub_feature_path1, sub_feature_path2, sub_feature_path3, model_dir, result_dir, feature_size1, feature_size2, feature_size3, feature_size4, label_size, model_number

lstm_unit = 256
full_connect_number = 1024
drop_prob = 0.8
learning_rate = 0.001
index_list = [1, 1, 1]
model_index = 4
attention_dim = 64
test_file = sys.argv[1]


class LSTM(object):

    def __init__(self, round_time, model_type):

        self.round_time = round_time

        if(model_type == "PDNA-543"):
                self.current_model_dir = model_dir + "/PDNA-543/" + str(self.round_time) + "/"
        else:
            self.current_model_dir = model_dir + "/PDNA-335/" + str(self.round_time) + "/"

        self.rewrite_single(self.current_model_dir, model_index)

    def rewrite_single(self, modeldir, index):

        model_name = "model" + str(index)
        f = open(modeldir + "/checkpoint", "w")
        f.write("model_checkpoint_path:" + "\"" + model_name + "\"" + "\n")
        f.write("all_model_checkpoint_paths:" + "\"" + model_name + "\"" + "\n")
        f.flush()
        f.close()

    def double_LSTM(self, x, keep_prob, name1, name2):  # double direction LSTM

        cell_fw_lstm_cells = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_unit, name = name1), output_keep_prob = keep_prob)
        cell_bw_lstm_cells = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_unit, name = name2), output_keep_prob = keep_prob)

        initial_state_fw = cell_fw_lstm_cells.zero_state(tf.shape(x)[0], dtype=tf.float32)
        initial_state_bw = cell_bw_lstm_cells.zero_state(tf.shape(x)[0], dtype=tf.float32)

        rnn_out, states = tf.nn.bidirectional_dynamic_rnn(cell_fw_lstm_cells, cell_bw_lstm_cells, inputs = x, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw)

        rnn_out = tf.concat([rnn_out[0], rnn_out[1]], 2)

        return rnn_out

    def weight_variable(self, shape, weight_name):

        initial = tf.truncated_normal(shape, stddev=0.1)
        weight = tf.Variable(initial, name = weight_name)
        return weight

    def self_attention(self, x, keep_prob, is_train):

        Q = tf.layers.dense(inputs = x, units = attention_dim)
        K = tf.layers.dense(inputs = x, units = attention_dim)
        V = tf.layers.dense(inputs = x, units = attention_dim)

        W = tf.matmul(Q, tf.transpose(K))
        W = tf.nn.softmax(W/math.sqrt(attention_dim*20))

        x = tf.matmul(W, V)
        x = tf.nn.dropout(x, keep_prob)

        return x


    def deepnn(self, x1, x2, x3, x4, y_, weight, keep_prob, is_train):  # deep CNN

        if (index_list[0] == 0):
            if (index_list[1] == 0):
                x = x3
            else:
                x = x2
                if (index_list[2] == 1):
                    x = tf.concat([x, x3], 1)
        else:
            x = x1

            if (index_list[1] == 1):
                x = tf.concat([x, x2], 1)
            if (index_list[2] == 1):
                x = tf.concat([x, x3], 1)

        x = tf.expand_dims(x, axis=0)

        with tf.name_scope('LSTM'):  # average

            x = self.double_LSTM(x, keep_prob, "lstm1", "lstm2")

        x = tf.squeeze(x, axis=0)

        x_final = self.self_attention(x, keep_prob, is_train)
        for i in range(9):
            tf.concat([x_final, self.self_attention(x, keep_prob, is_train)], axis=1)

        x = x_final

        with tf.name_scope('fc'):  # fully connected layer

            x = tf.layers.dense(inputs = x, units = full_connect_number, activation = tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        embeddings = tf.nn.l2_normalize(x, axis=1)

        with tf.name_scope('output'):  # output layer

            probability = tf.layers.dense(inputs = x, units = label_size, activation = tf.nn.softmax)

        with tf.name_scope('caculate_loss'):   # loss function

            weight = tf.expand_dims(weight, axis=1)

            cross_entropy = y_ * tf.log(probability + 1e-6)
            cross_entropy = -tf.reduce_sum(cross_entropy)

        with tf.name_scope('adam_optimizer'):  # optimization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        return train_step, probability, cross_entropy, embeddings

    def running(self): #running process

        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, feature_size1])
        x2 = tf.placeholder(tf.float32, [None, feature_size2])
        x3 = tf.placeholder(tf.float32, [None, feature_size3])
        x4 = tf.placeholder(tf.float32, [None, feature_size4])
        y_ = tf.placeholder(tf.float32, [None, label_size])
        weight = tf.placeholder(tf.float32, [None])

        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)

        train_step, probability, cross_entropy, embeddings  = self.deepnn(x1, x2, x3, x4, y_, weight, keep_prob, is_train)


        with tf.Session() as sess:

            ckpt = tf.train.latest_checkpoint(self.current_model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

            name_list = rd.get_name_list(test_file)

            current_result_dir = result_dir + "/" + str(self.round_time) + "/"
            os.system("rm -rf " + current_result_dir)
            os.makedirs(current_result_dir)

            for name in name_list:

                feature1, feature2, feature3, feature4, label, test_weight = rd.read_data_single(sub_feature_path1, sub_feature_path2, sub_feature_path3, name)
                print(feature1.shape)
                print(feature2.shape)
                print(feature3.shape)
                test_score = sess.run(probability, feed_dict={x1: feature1, x2: feature2, x3: feature3, x4: feature4, y_: label, weight: test_weight, keep_prob: drop_prob, is_train: False})

                result_file = current_result_dir + "/" + name
                f = open(result_file, "w")
                for i in range(test_score.shape[0]):
                    f.write(str(test_score[i][0]) + "\n")
                f.close()

if __name__ == '__main__':

    model_type = sys.argv[2]

    for round_time in range(1, model_number+1):
        lstm = LSTM(round_time, model_type)
        lstm.running()










