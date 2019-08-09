import numpy as np

import keras
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, concatenate, Flatten, Dropout
from keras.constraints import maxnorm
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate_music import evaluate_model
from Dataset_music import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
from keras.engine.topology import Layer

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='music',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()



class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        Q_seq, K_seq, V_seq = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)



def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=100, name='user_embedding',
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                          seed=None),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=100, name='item_embedding',
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                          seed=None),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    # user_latent = Flatten()(MLP_Embedding_User(user_input))
    # item_latent = Flatten()(MLP_Embedding_Item(item_input))
    user_latent = MLP_Embedding_User(user_input)
    item_latent = MLP_Embedding_Item(item_input)
    user_latent = Reshape((10, 10))(user_latent)
    item_latent = Reshape((10, 10))(item_latent)

    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_latent, item_latent], axis=1)
    O_seq = Attention(6, 80)([vector, vector, vector])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(1, activation='sigmoid')(O_seq)

    model = Model(inputs=[user_input, item_input],
                  outputs=outputs)

    return model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        if train[(u,i)]==1:
            labels.append(1)
        if train[(u,i)]==-1:
            labels.append(0)
        '''
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
        '''
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 1
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        # Check Init performance
    #t1 = time()
    #(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    #hr, ndcg= np.array(hits).mean(), np.array(ndcgs).mean()
    #print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_ap, best_iter = 0, 0, 0, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # EvaluationD
        if epoch % verbose == 0:
            h3s, h5s, n3s, n5s, a3s, a5s, h1s, rrs = evaluate_model(model, testRatings, testNegatives, 3,
                                                                    evaluation_threads)
            h3, h5, n3, n5, a3, a5, h1, rr, loss = np.array(h3s).mean(), np.array(h5s).mean(), np.array(
                n3s).mean(), np.array(n5s).mean(), np.array(a3s).mean(), np.array(
                a5s).mean(), np.array(h1s).mean(), np.array(rrs).mean(), hist.history['loss'][0]
            print(
                'Iteration %d [%.1f s]: HR3 = %.4f, HR5 = %.4f, N3 = %.4f, N5 = %.4f,  A3 = %.4f, A5 = %.4f, loss = %.4f [%.1f s]'
                % (epoch, t2 - t1, h3, h5, n3, n5, a3, a5, loss, time() - t2))
            if h3 > best_hr:
                best_hr, best_h5, best_ndcg, best_n5, best_ap, best_a5, best_h1, best_rr, best_iter = h3, h5, n3, n5, a3, a5, h1, rr, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f. " % (
        best_iter, best_hr, best_h5, best_ndcg, best_n5, best_ap, best_a5, best_h1, best_rr))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
    file_name = "SA(amazon).txt"
    f = open(file_name, 'a+')
    content = str(best_hr) + '\t' + str(best_h5) + '\t' + str(best_ndcg) + '\t' + str(
        best_n5) + '\t' + \
              str(best_ap) + '\t' + str(best_a5) + '\n'
    f.write(content)
    f.close()