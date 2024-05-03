import math
import sys
import os
import Read_Data1 as rd
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]
gpu_ratio = float(sys.argv[4])
import Calculate_results as cr

class LSTM(object):

    def __init__(self, workdir, feature_dir1, feature_dir2, feature_dir3, feature_dir4,
                 feature_size1, feature_size2, feature_size3, feature_size4, label_size,
                 lstm_unit, full_connect_number, drop_prob, learning_rate, max_iteration, round_time, index_list):

        self.workdir = workdir
        self.feature_dir1 = feature_dir1
        self.feature_dir2 = feature_dir2
        self.feature_dir3 = feature_dir3
        self.feature_dir4 = feature_dir4

        self.feature_size1 = feature_size1
        self.feature_size2 = feature_size2
        self.feature_size3 = feature_size3
        self.feature_size4 = feature_size4
        self.label_size = label_size

        self.lstm_unit = lstm_unit
        self.full_connect_number = full_connect_number
        self.drop_prob = drop_prob
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.round_time = round_time

        self.index_list = index_list
        self.postfix = ""
        for index in self.index_list:
            self.postfix = self.postfix + str(index)

        self.recordfile = workdir + "/record_loss_" + str(round_time)
        self.model_dir = workdir + "/model" + self.postfix + "/" + str(round_time) + "/"

        if (os.path.exists(self.model_dir) == False):
            os.makedirs(self.model_dir)

    def double_LSTM(self, x, keep_prob, name1, name2):  # double direction LSTM

        cell_fw_lstm_cells = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_unit, name = name1), output_keep_prob = keep_prob)
        cell_bw_lstm_cells = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.lstm_unit, name = name2), output_keep_prob = keep_prob)

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

        attention_dim = 64

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

        #x_final = self.self_attention(x, keep_prob, is_train)

        x = tf.expand_dims(x, axis=0)

        with tf.name_scope('LSTM'):  # average

            x = self.double_LSTM(x, keep_prob, "lstm1", "lstm2")

        x = tf.squeeze(x, axis=0)

        x_final = self.self_attention(x, keep_prob, is_train)
        for i in range(9):
            tf.concat([x_final, self.self_attention(x, keep_prob, is_train)], axis=1)

        x = x_final

        with tf.name_scope('fc'):  # fully connected layer

            x = tf.layers.dense(inputs = x, units = self.full_connect_number, activation = tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        embeddings = tf.nn.l2_normalize(x, axis=1)

        with tf.name_scope('output'):  # output layer

            probability = tf.layers.dense(inputs = x, units = self.label_size, activation = tf.nn.softmax)

        with tf.name_scope('caculate_loss'):   # loss function

            weight = tf.expand_dims(weight, axis=1)

            cross_entropy = y_ * tf.log(probability + 1e-6)
            cross_entropy = -tf.reduce_sum(cross_entropy)

        with tf.name_scope('adam_optimizer'):  # optimization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        return train_step, probability, cross_entropy, embeddings

    def running(self): #running process

        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, self.feature_size1])
        x2 = tf.placeholder(tf.float32, [None, self.feature_size2])
        x3 = tf.placeholder(tf.float32, [None, self.feature_size3])
        x4 = tf.placeholder(tf.float32, [None, self.feature_size4])
        y_ = tf.placeholder(tf.float32, [None, self.label_size])
        weight = tf.placeholder(tf.float32, [None])

        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)

        train_step, probability, cross_entropy, embeddings  = self.deepnn(x1, x2, x3, x4, y_, weight, keep_prob, is_train)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_ratio

        with tf.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())

            f = open(self.recordfile, "w")

            for iteration in range(1, self.max_iteration + 1):

                batch_train_name_list, train_sequence_dict, train_label_dict = rd.create_data_set(self.workdir, "train", True)
                batch_test_name_list, test_sequence_dict, test_label_dict = rd.create_data_set(self.workdir, "test", False)

                resultdir = workdir + "/cross_entropy_" + self.postfix + "_result/test/round" + str(self.round_time) + "/" + str(iteration) + "/"
                print(resultdir)
                os.system("rm -rf " + resultdir)



                # training
                for i in range(len(batch_train_name_list)):
                    train_name, train_feature1, train_feature2, train_feature3, train_feature4, train_label, train_weight = rd.read_data_single(self.feature_dir1, self.feature_dir2, self.feature_dir3, self.feature_dir4, batch_train_name_list[i], train_sequence_dict, train_label_dict)
                    train_step.run(feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, x4: train_feature4, y_: train_label, weight: train_weight, keep_prob: self.drop_prob, is_train: True})
                    print(i)

                    #my_weight = sess.run(attention_weight, feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, x4: train_feature4, y_: train_label, weight: train_weight, keep_prob: self.drop_prob, is_train: True})
                    #print(my_weight)
                    #print(np.sum(my_weight, axis = 1))

                train_loss = 0

                '''

                for i in range(len(batch_train_name_list)):
                    train_name, train_feature1, train_feature2, train_feature3, train_feature4, train_label, train_weight = rd.read_data_single(self.feature_dir1, self.feature_dir2, self.feature_dir3, self.feature_dir4, batch_train_name_list[i], train_label_dict)
                    train_loss = train_loss + sess.run(cross_entropy, feed_dict={x1: train_feature1, x2: train_feature2, x3: train_feature3, x4: train_feature4, y_: train_label, weight: train_weight, keep_prob: self.drop_prob, is_train: True})
                '''

                f.write("The " + str(iteration) + "-th iteration: \ntraining loss=" + str(train_loss) + "\n")
                f.flush()

                # test

                test_loss = 0;

                for i in range(len(batch_test_name_list)):
                    test_name, test_feature1, test_feature2, test_feature3, test_feature4, test_label, test_weight = rd.read_data_single(self.feature_dir1, self.feature_dir2, self.feature_dir3, self.feature_dir4, batch_test_name_list[i], test_sequence_dict, test_label_dict)
                    test_loss = test_loss + sess.run(cross_entropy, feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3, x4: test_feature4, y_: test_label, weight: test_weight, keep_prob: self.drop_prob, is_train: False})
                    test_score = sess.run(probability, feed_dict={x1: test_feature1, x2: test_feature2, x3: test_feature3, x4: test_feature4, y_: test_label, weight: test_weight, keep_prob: self.drop_prob, is_train: False})
                    cr.save_results(self.workdir, "cross_entropy_" + self.postfix, "test", self.round_time, iteration, test_name, test_score, test_label)

                f.write("test loss=" + str(test_loss) + "\n")
                f.flush()

                # save model
                #saver = tf.train.Saver(max_to_keep=0)
                #saver.save(sess, self.model_dir + "model" + str(iteration))

                cr.create_cross_entropy_result(self.workdir, "cross_entropy_" + self.postfix, "test", self.round_time, iteration, self.postfix)

            f.close()

if __name__ == '__main__':

    workdir = sys.argv[1]
    round_time = int(sys.argv[2])

    feature_dir1 = "/data1/zhuyiheng/DNA/features/"
    feature_dir2 = "/data1/zhuyiheng/DNA/features_protrans/"
    feature_dir3 = "/data1/zhuyiheng/DNA/msa_features_256/"
    feature_dir4 = "/data1/zhuyiheng/DNA/dssp_feature/"

    feature_size1 = 2560
    feature_size2 = 1024
    feature_size3 = 768
    feature_size4 = 18
    label_size = 2

    lstm_unit = 256
    full_connect_number = 1024
    drop_prob = 0.8
    learning_rate = 0.001
    max_iteration = int(sys.argv[5])

    index_list = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])]

    cn = LSTM(workdir, feature_dir1, feature_dir2, feature_dir3, feature_dir4,
              feature_size1, feature_size2, feature_size3, feature_size4, label_size,
              lstm_unit, full_connect_number, drop_prob, learning_rate, max_iteration, round_time, index_list)
    cn.running()









