import numpy as np
import keras
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, concatenate, Flatten, Dropout, multiply
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='bookcross',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=0,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def get_model(num_users, num_items, layers=[10], reg_layers=[0], reg_mf=0, mf_dim=10):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                         seed=None),
                                  embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                          seed=None),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05,
                                                                                          seed=None),
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = multiply([mf_user_latent, mf_item_latent])  # element-wise multiply

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

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
        # negative instances
        '''
        
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

    dataset="bookcross"
    args = parse_args()
    path = args.path
    d_temp = dataset
    #dataset = "douban" + dataset
    temp_d = dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose


    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(dataset)
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
    t1 = time()
    #(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    #hr, ndcg= np.array(hits).mean(), np.array(ndcgs).mean()
    #print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_ap, best_iter, best_h5 = 0, 0, 0, -1, 0
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
            h3s, h5s, h8s, n3s, n5s, n8s, a3s, a5s, a8s, rrs = evaluate_model(model, testRatings, testNegatives, 3,
                                                                         evaluation_threads)
            h3, h5, h8, n3, n5, n8, a3, a5, a8, rr, loss = np.array(h3s).mean(), np.array(h5s).mean(), np.array(
                h8s).mean(), np.array(
                n3s).mean(), np.array(n5s).mean(), np.array(n8s).mean(), np.array(a3s).mean(), np.array(
                a5s).mean(), np.array(a8s).mean(), np.array(rrs).mean(), hist.history['loss'][0]
            print(
                'Iteration %d [%.1f s]: HR1 = %.4f, HR3 = %.4f, HR5 = %.4f, N1 = %.4f, N3 = %.4f, N5 = %.4f, A1 = %.4f, A3 = %.4f, A5 = %.4f, loss = %.4f [%.1f s]'
                % (epoch, t2 - t1, h3, h5, h8, n3, n5, n8, a3, a5, a8, loss, time() - t2))
            if h5 > best_h5:
                best_hr, best_h5, best_h8, best_ndcg, best_n5, best_n8, best_ap, best_a5, best_a8, best_rr, best_iter = h3, h5, h8, n3, n5, n8, a3, a5, a8, rr,epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f. " % (
        best_iter, best_hr, best_h5, best_h8, best_ndcg, best_n5, best_n8, best_ap, best_a5, best_a8, best_rr))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))

