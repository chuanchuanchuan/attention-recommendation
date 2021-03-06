import numpy as np

import keras
from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation, Reshape
from keras.layers import Embedding, Input, Dense, merge, concatenate, Flatten, Dropout, Permute
from keras.constraints import maxnorm
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
from keras.engine.topology import Layer
import pickle
import math
import heapq  # for retrieval topK
import multiprocessing
from time import time

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)


_model = None
_testRatings = None
_testNegatives = None
_K = None



def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits3, hits5, hits8, ndcg3, ndcg5, ndcg8, ap3, ap5, ap8, r = [], [], [], [], [], [], [], [], [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        hr, ndcg, ap, rr = eval_one_rating(idx)
        hits3.append(hr[0])
        hits5.append(hr[1])
        hits8.append(hr[2])
        ndcg3.append(ndcg[0])
        ndcg5.append(ndcg[1])
        ndcg8.append(ndcg[2])
        ap3.append(ap[0])
        ap5.append(ap[1])
        ap8.append(ap[2])
        r.append(rr)
    return hits3, hits5, hits8, ndcg3, ndcg5, ndcg8, ap3, ap5, ap8, r


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    for i in range(5):
        items.append(gtItem[i])
    # Get prediction scores
    map_item_score = {}
    user_input = []
    item_input = []

    ul = list(user_info[u])

    for i in range(len(items)):
        user_input.append(ul)
    for i in items:
        item_input.append(movie_info[i])
    predictions = _model.predict([np.array(user_input), np.array(item_input)],
                                 batch_size=64, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(8, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    ap = getAP(ranklist, gtItem)
    rr = getRR(ranklist, gtItem)
    return hr, ndcg, ap, rr


def getHitRatio(ranklist, gtItem):
    result = []
    for i in [1, 3, 5]:
        count = 0
        for item in ranklist[0:i]:
            if item in gtItem:
                count += 1
        result.append(count / i)
    return result


def getNDCG(ranklist, gtItem):
    result = []
    for j in [1, 3, 5]:
        count = 0
        idcg = 0
        for i in range(j):
            item = ranklist[i]
            if item in gtItem:
                count += math.log(2) / math.log(i + 2)
        for i in range(j):
            idcg += math.log(2) / math.log(i + 2)
        result.append(count / idcg)
    return result


def getAP(ranklist, gtItem):
    result = []
    for j in [1, 3, 5]:
        count = 0
        p = []
        for i in range(j):
            item = ranklist[i]
            if item in gtItem:
                count += 1
                p.append(count / (i + 1))
        if len(p)==0:
            result.append(0)
        else:
            result.append(np.sum(p)/j)
    return result

def getRR(ranklist, gtItem):
    for i in range(5):
        item = ranklist[i]
        if item in gtItem:
            return 1/(i+1)
    return 0

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='bookcross',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
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



def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(3,), name='user_input')
    item_input = Input(shape=(3,), name='item_input')

    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_input, item_input])
    vector = Embedding(input_dim=vocabulary_lenth + 1, output_dim=100)(vector)
    vector = Flatten()(vector)

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

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
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


if __name__ == '__main__':
    dataset = "bookcross"
    # vocabulary_lenth = 85124
    # vocabulary_lenth = 28343
    vocabulary_lenth = 35124

    with open("user_hist_" + dataset, "rb") as file:
        user_info = pickle.load(file)

    with open("book_info_" + dataset, "rb") as file:
        movie_info = pickle.load(file)
    args = parse_args()
    path = args.path
    path=''
    #dataset = args.dataset
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
    model_out_file = 'Pretrain/savedmodel'

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    #model.load_weights('Pretrain/savedmodel')
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
    #hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    #print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    #best_hr, best_ndcg, best_iter = hr, ndcg, -1
    best_hr, best_ndcg, best_iter, best_h5 = 0, 0, -1, 0
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        userinput = []
        iteminput = []
        for uid in user_input:
            userinput.append(list(user_info[uid]))
        for iid in item_input:
            iteminput.append(list(movie_info[iid]))

        # Training
        hist = model.fit([np.array(userinput), np.array(iteminput)],  # input
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
                best_hr, best_h5, best_h8, best_ndcg, best_n5, best_n8, best_ap, best_a5, best_a8, best_rr, best_iter = h3, h5, h8, n3, n5, n8, a3, a5, a8, rr, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f. " % (
        best_iter, best_hr, best_h5, best_h8, best_ndcg, best_n5, best_n8, best_ap, best_a5, best_a8, best_rr))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
