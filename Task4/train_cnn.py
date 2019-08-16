#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2019/7/23 下午5:12
@Author : mia
"""
import pickle
import os
import tensorflow as tf
import random
import numpy
from lstm_matching_model import LstmMatchingModel
import sys
import codecs
import re
import numpy as np
from TextCNN_tf import TextCNN
import time


def clean_str(s):
    """Clean sentence"""
    s = re.sub("\(", "（", s)
    s = re.sub("\)", "）", s)
    s = re.sub(" ", "", s)
    s = re.sub("\t", "", s)

    return s.strip().lower()


def load_data(data_dir):
    dataset = []
    len_list = []
    with open(data_dir, "r") as f:
        content = f.readlines()
        for line in content:
            line = line.rstrip("\n")
            line = line.strip(' ')
            splits = line.split("\t")

            if not len(splits) == 2:
                print(line)
                print('line wrong')
                exit()
            data = {}

            if splits[0]:
                data["label"] = splits[0].split(' ')

            # TODO feature可以改
            feature_num = 1

            for index in range(feature_num):
                str_raw = clean_str(splits[index + 1])
                print(str_raw)
                # str_raw = str_raw.lower() # 英文转小写
                str_list = []
                aj_list = []
                for aj, a in enumerate(str_raw):
                    if aj < len(str_raw) - 2 and a == 'n' and str_raw[aj + 1] == 'a' and str_raw[aj + 2] == 'n':
                        str_list.append('nan')
                        aj_list.append(aj)
                        aj_list.append(aj + 1)
                        aj_list.append(aj + 2)

                    elif aj < len(str_raw) - 4 and a == '@' and str_raw[aj + 1] == '@' and str_raw[aj + 2] == '@' \
                            and str_raw[aj + 3] == '@':
                        str_list.append('@@@@')
                        aj_list.append(aj)
                        aj_list.append(aj + 1)
                        aj_list.append(aj + 2)
                        aj_list.append(aj + 3)

                    if not aj in aj_list:
                        str_list.append(a)

                data["feature_%s" %index] = str_list
                if index == 0:
                    len_list.append(len(str_list) + 2)
                # print(data)
            dataset.append(data)
    # print(max(len_list))
    # print(len_list)
    # print(len(dataset))
    return dataset


def load_map(dataset, id_folder, isTrain):
    if not isTrain:
        with open(os.path.join(id_folder, "maps.pkl"), "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
            return char_to_id, id_to_char, tag_to_id, id_to_tag
    char_id = 0
    tag_id = 0
    char_to_id = {}
    id_to_char = []
    tag_to_id = {}
    id_to_tag = []
    for data in dataset:
        feature_num = 1
        for index in range(feature_num):
            for char in data['feature_%s'% index]:
                if not char in char_to_id:
                    char_to_id[char] = char_id
                    id_to_char.append(char)
                    char_id += 1

        if 'label' in data:
            for tag in data['label']:
                if not tag in tag_to_id:
                    tag_to_id[tag] = tag_id
                    id_to_tag.append(tag)
                    tag_id += 1
    char_to_id["[[[UNK]]]"] = char_id
    id_to_char.append("[[[UNK]]]")
    char_id += 1
    if not os.path.exists(id_folder):
        os.mkdir(id_folder)

    with open(os.path.join(id_folder, "maps.pkl"), "wb") as f:
        pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

    print(char_to_id)
    print(id_to_char)
    print(len(id_to_char))
    print(tag_to_id)
    print(id_to_tag)
    print(len(id_to_tag))

    return char_to_id, id_to_char, tag_to_id, id_to_tag


def prepare_data(raw_dataset, char_to_id, tag_to_id):
    dataset = {}
    dataset['label'] = []
    feature_num = 1
    for index in range(feature_num):
        dataset["feature_%s"% index] = []
        dataset["end_pos_%s"% index] = []

    for data in raw_dataset:
        for index in range(feature_num):
            features = []
            for char in data["feature_%s" % index]:
                if char in char_to_id:
                    features.append(char_to_id[char])
                else:
                    features.append(char_to_id["[[[UNK]]]"])

                if len(features) >= 100: # max_len
                    break
            t = len(features)

            for i in range(t, 100):
                features.append(len(char_to_id) - 1)
            dataset["feature_%s" % index].append(features)
            dataset["end_pos_%s" % index].append(t)

        labels = [0.0 for x in range(len(tag_to_id))]
        if 'label' in data:
            for tag in data["label"]:
                if tag in tag_to_id:
                    id = tag_to_id[tag]
                    labels[id] = 1.0
        dataset["label"].append(labels)

    return dataset


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    Load word embedding from pre-trained file
    embedding size must match
    """
    print("old_weights")
    print(old_weights)
    new_weights = old_weights
    print("new_weights")
    print(new_weights)
    sys.stderr.write('Loading pretrained embeddings from {}...\n'.format(emb_path))
    pre_trained = {}
    print('pre_trained')
    print(pre_trained)
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        print(len(line))
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = numpy.array([float(x) for x in line[1:]]).astype(numpy.float16)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        sys.stderr.write('WARNING: %i invalid lines\n' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    print(n_words)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        print('word')
        print(word)
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
        # else:
        #    print(word)
    sys.stderr.write('Loaded %i pretrained embeddings.\n' % len(pre_trained))
    sys.stderr.write('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.\n' % (
        c_found + c_lower + c_zeros, n_words,
        100. * (c_found + c_lower + c_zeros) / n_words)
    )
    sys.stderr.write('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.\n' % (c_found, c_lower, c_zeros))
    return new_weights


def test(test_dataset, parameters, model, sess):
    batch_size = parameters["batch_size"]
    feature_num = parameters['feature_num']
    total_batch = float(len(test_dataset["label"])) / batch_size
    import math
    total_batch = int(math.ceil(total_batch))
    sum_good = [0 for x in range(parameters["label_sum"] + 1)]
    sum_pred = [0 for x in range(parameters["label_sum"] + 1)]
    sum_oracle = [0 for x in range(parameters["label_sum"] + 1)]
    # Loop over all batches
    for i in range(total_batch):
        batch_xs_list = []
        batch_add_list = []
        batch_end_list = []
        for _ in range(feature_num):
            batch_xs_list.append([])
            batch_add_list.append([])
            batch_end_list.append([])

        batch_ys = test_dataset["label"][(i * batch_size):((i + 1) * batch_size)]
        for index_l in range(0, feature_num):
            batch_xs_list[index_l] = (test_dataset["feature_%s" % index_l][(i * batch_size):((i + 1) * batch_size)])
            # batch_add_list[index_l] = (test_dataset["add_%s" % index_l][(i * batch_size):((i + 1) * batch_size)])
            batch_end_list[index_l] = (test_dataset["end_pos_%s" % index_l][(i * batch_size):((i + 1) * batch_size)])

        pred = sess.run([model.pred], feed_dict={model.x_0: batch_xs_list[0],
                                                 model.y: batch_ys,
                                                 model.end_pos_0: batch_end_list[0],
                                                 model.dropout: 1.0})
        oracle = batch_ys
        for t in range(len(pred)):
            for batch in range(len(pred[t])):
                for i in range(parameters["label_sum"] + 1):
                    if i == parameters['label_sum']:
                        if (pred[t][batch] < 0.5).all():
                            sum_pred[i] += 1
                        if (np.array(oracle[batch]) == 0.0).all():
                            sum_oracle[i] += 1
                        if (pred[t][batch] < 0.5).all() and (np.array(oracle[batch]) == 0.0).all():
                            sum_good[i] += 1
                        break

                    pred_bool = (pred[t][batch][i] >= 0.5)
                    oracle_bool = (oracle[batch][i] == 1.0)
                    if pred_bool:
                        sum_pred[i] += 1
                    if oracle_bool:
                        sum_oracle[i] += 1
                    if pred_bool and oracle_bool:
                        sum_good[i] += 1

    for i in range(parameters["label_sum"] + 1):
        print("label_%i precision[%.2f] recall[%.2f] oracle_sum[%i]" % (
        i, sum_good[i] * 100.0 / sum_pred[i] if sum_pred[i] > 0 else 0.0,
        sum_good[i] * 100.0 / sum_oracle[i] if sum_oracle[i] > 0 else 0.0,
        sum_oracle[i]))
    print("label_all precision[%.2f] recall[%.2f] oracle_sum[%i]" % (
    sum(sum_good) * 100.0 / sum(sum_pred) if sum(sum_pred) > 0 else 0.0,
    sum(sum_good) * 100.0 / sum(sum_oracle) if sum(sum_oracle) > 0 else 0.0,
    sum(sum_oracle)))
    sys.stdout.flush()
    return (sum(sum_good) * 100.0 / sum(sum_pred) if sum(sum_pred) > 0 else 0.0), (
        sum(sum_good) * 100.0 / sum(sum_oracle) if sum(sum_oracle) > 0 else 0.0)


def train(train_dataset, test_dataset, parameters):
    tf.set_random_seed(1234)
    random.seed(50)
    numpy.random.seed(123)
    training_epochs = 100
    learning_rate = 0.01
    batch_size = 128
    feature_num = 1
    display_step = 1
    cnn = TextCNN(parameters)
    # 定义训练过程
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 指定优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_vars = optimizer.compute_gradients(cnn.loss)
    capped_grads_vars = [[g, v] if g is None else [tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_vars]
    train_op = optimizer.apply_gradients(capped_grads_vars, global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)


    pre_prec = 0.0
    pre_recall = 0.0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(init)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

            # Generate batches
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...开始训练了
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, train_op)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    train_raw_dataset = load_data('data/20190723科目名整理.txt')

    char_to_id, id_to_char, tag_to_id, id_to_tag = load_map(train_raw_dataset, id_folder='./id_folder', isTrain=True)

    print('char_to_id : %s' %char_to_id)
    print('id_to_char: %s' %id_to_char)
    print(tag_to_id)
    print(id_to_tag)
    train_data = prepare_data(train_raw_dataset, char_to_id, tag_to_id)
    with open('data/label.txt', 'w') as f:
        for i in id_to_tag:
            f.write(i)
            f.write('\n')

    parameters = {
                     "embedding_size": 128,
                     "sentence_len": 50,
                     "save_path": "model",
                     "hidden_dim": 200,
                     "batch_size": 64,
                     "dropout": 0.8,
                     "feature_num": 1,
                     "add_num": 0,
                     "num_classes": len(id_to_tag),
                     "vocab_size": len(id_to_char),
                     "filter_sizes": "2,3,4",
                     "num_filters": 64,
                     "l2_reg_lambda": 0
    }
    with open(os.path.join('./id_folder',"parameters.pkl"), "wb") as f:
        pickle.dump([parameters], f)

    # train(train_data, train_data, parameters)
