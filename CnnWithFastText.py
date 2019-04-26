# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:12:50 2019

@author: hadeer
"""

import numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from nltk import word_tokenize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support


def load_comment_data(filename, max_length):
    df = pd.read_csv("G:\\level4\\smester 2\\nlp\\project\\rscience-popular-comment-removal\\" + filename, encoding = "ISO-8859-1")
    df = df.sample(frac=1.0)
    data = df.values
    x = data[:,2:3]
    x = [i[0] for i in x]
    y = data[:,3:]
    temp = []
    lengths = []
    skipped = []
    maxi = 0
    for i in range(len(x)):
        # Remove ' from text as fasttext does not seem to embed "don't" but does have "dont"
        comment = x[i].lower().replace('\'', '')
        string_arr = word_tokenize(comment)
        length = len(string_arr)
        if length > maxi:
            maxi = length
        string_arr = string_arr[:max_length]
        if length < 1:
            skipped.append(i)
            continue
        lengths.append(length)
        temp.append(string_arr)

    x = temp
    temp = []
    for i in range(len(y)):
        if i in skipped:
            continue
        label = y[i]
        if label:
            temp.append([0, 1])
        else:
            temp.append([1, 0])

    y = temp
    return x, np.array(y), np.array(lengths)

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

max_length = 200
train_x, train_y, train_lengths = load_comment_data('reddit_train.csv', max_length)
largest_length = max(train_lengths)
test_x, test_y, test_lengths = load_comment_data('reddit_test.csv', max_length)
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for i,o in enumerate(open("G:\\level4\\smester 2\\nlp\\project\\wiki-news-300d-1M-subword.vec\\wiki-news-300d-1M-subword.vec", encoding="utf8")))

def word_embed(sentences, max_length):
    vector_x = []
    for sentence in sentences:
        sentence_x = []
        for word in sentence:
            if word in embeddings_index:
                vector = embeddings_index[word]
            else:
                vector = np.array([0.0] * 300)
            sentence_x.append(vector)

        diff = max_length - len(sentence_x)
        zeros = np.zeros((diff, 300))
        sentence_x = np.array(sentence_x)
        if len(sentence_x.shape) < 2:
            print(sentence_x)
        if sentence_x.shape[1] != zeros.shape[1]:
            print(sentence_x.shape, zeros.shape)
        sentence_x = np.append(sentence_x, zeros, axis=0)
        vector_x.append(sentence_x)
    
    return np.array(vector_x)

validation_x = train_x[-3000:]
vector_validation = word_embed(validation_x, max_length)
validation_y = train_y[-3000:]
validation_lengths = train_lengths[-3000:]
train_x = train_x[:-3000]
train_y = train_y[:-3000]
train_lengths = train_lengths[:-3000]

tf.reset_default_graph()
inputs = tf.placeholder(shape=(None, max_length, 300), dtype=tf.float32)
expanded_inputs = tf.expand_dims(inputs, 3)
keep_prob = tf.placeholder(shape=(), dtype=tf.float32)
keep_prob_conv = tf.placeholder(shape=(), dtype=tf.float32)
inputs_length = tf.placeholder(shape=(None, ), dtype=tf.float32)
lengths = tf.expand_dims(inputs_length, 1)
targets = tf.placeholder(shape=(None, 2), dtype=tf.float32)
conv_layer2 = tf.layers.conv2d(expanded_inputs, 16, (4, 300), (1, 1), activation=tf.nn.relu)
conv_layer2 = tf.nn.dropout(conv_layer2, keep_prob=keep_prob_conv)
conv_layer4 = tf.layers.conv2d(expanded_inputs, 16, (6, 300), (1, 1), activation=tf.nn.relu)
conv_layer4 = tf.nn.dropout(conv_layer4, keep_prob=keep_prob_conv)
conv_layer6 = tf.layers.conv2d(expanded_inputs, 16, (8, 300), (1, 1), activation=tf.nn.relu)
conv_layer6 = tf.nn.dropout(conv_layer6, keep_prob=keep_prob_conv)
squeeze2 = tf.squeeze(conv_layer2, 2)
squeeze4 = tf.squeeze(conv_layer4, 2)
squeeze6 = tf.squeeze(conv_layer6, 2)
pool2 = tf.layers.max_pooling1d(squeeze2, max_length-4+1, 1)
pool4 = tf.layers.max_pooling1d(squeeze4, max_length-6+1, 1)
pool6 = tf.layers.max_pooling1d(squeeze6, max_length-8+1, 1)
pools = [pool2, pool4, pool6]
pools = [tf.squeeze(x, 1) for x in pools]
pools.append(lengths)
concat_layers = tf.concat(pools, axis=1)
hidden_layer = tf.layers.dense(concat_layers, 32, activation=tf.nn.relu)
final_layer = tf.layers.dense(hidden_layer, 2, activation=tf.nn.softmax)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(targets, 1), tf.math.argmax(final_layer,1)), tf.float32))
optimizer = tf.train.AdamOptimizer(0.001)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=final_layer))
var = tf.trainable_variables() 
loss_l2 = tf.add_n([ tf.nn.l2_loss(v) for v in var
                    if 'bias' not in v.name ]) * 0.001
training_op = optimizer.minimize(loss + loss_l2)

test_batch_size = 1000
batch_size = 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(5):
        avg_train_error = []
        avg_train_acc = []
        train_acc = 0
        t0 = time.time()
        for i in range((len(train_x) // batch_size) + 1):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            vector_x = word_embed(batch_x, max_length)
            batch_lengths = train_lengths[i*batch_size:(i+1)*batch_size] / largest_length
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            feed = {inputs: vector_x, keep_prob: 0.9, keep_prob_conv: 0.5, inputs_length: batch_lengths,targets: batch_y}
            _, train_loss, train_acc = sess.run([training_op, loss, accuracy], feed)
            avg_train_error.append(train_loss)
            avg_train_acc.append(train_acc)

        feed = {inputs:vector_validation, keep_prob:1.0, keep_prob_conv:1.0, inputs_length:(validation_lengths / largest_length), targets:validation_y}
        validation_err, validation_acc = sess.run([loss, accuracy], feed)            
        avg_train_error = sum(avg_train_error)/len(avg_train_error)
        avg_train_acc = sum(avg_train_acc)/len(avg_train_acc)
        print("----------------------------------------------------------------")
        print("Epoch: " + str(epoch))
        print("Average error: " + str(avg_train_error))
        print("Average accuracy: " + str(avg_train_acc))
        print("Validation error: {}".format(validation_err))
        print("Validation accuray: {}".format(validation_acc))
        
    avg_test_err = []
    avg_test_acc = []
    outputs = np.array([])
    for i in range((len(test_y) // test_batch_size) + 1):
        batch_x = test_x[i*test_batch_size:(i+1)*test_batch_size]
        vector_x = word_embed(batch_x, max_length)
        batch_lengths = test_lengths[i*test_batch_size:(i+1)*test_batch_size] / largest_length
        batch_y = test_y[i*test_batch_size:(i+1)*test_batch_size]  
        feed = {inputs: vector_x, keep_prob: 1.0, keep_prob_conv: 1.0, inputs_length: batch_lengths, targets: batch_y}
        test_loss, test_acc, pred = sess.run([loss, accuracy, final_layer], feed)
        pred = pred[:,1:]
        outputs = np.append(outputs, pred)
        avg_test_err.append(test_loss)
        avg_test_acc.append(test_acc)
        
    auc_score = roc_auc_score(np.argmax(test_y, axis=1), outputs)
    recall_fscore= precision_recall_fscore_support(np.argmax(test_y,axis=1),outputs.round(), average='micro')
    print("----------------------------------------------------------------")
    print("Test error: {}".format(sum(avg_test_err)/len(avg_test_err)))
    print("Test accuray: {}".format(sum(avg_test_acc)/len(avg_test_acc)))
    print("AUC score: {}".format(auc_score))
    print("precision_recall_fscore: {}".format(recall_fscore))
    print("----------------------------------------------------------------")
    