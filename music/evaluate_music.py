import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time


# Global variables that are shared across processes
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

    hits3, hits5, ndcg3, ndcg5, ap3, ap5, h1, r= [], [], [], [], [], [], [], []
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
        ndcg3.append(ndcg[0])
        ndcg5.append(ndcg[1])
        ap3.append(ap[0])
        ap5.append(ap[1])
        h1.append(ap[2])
        r.append(rr)
    return hits3, hits5, ndcg3, ndcg5, ap3, ap5, h1, r


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    for i in range(5):
        items.append(gtItem[i])
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=64, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(5, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    ap = getAP(ranklist, gtItem)
    rr = getRR(ranklist,gtItem)
    return hr, ndcg, ap, rr


def getHitRatio(ranklist, gtItem):
    result = []
    for i in [3, 5]:
        count = 0
        for item in ranklist[0:i]:
            if item in gtItem:
                count += 1
        result.append(count / i)
    return result


def getNDCG(ranklist, gtItem):
    result = []
    for j in [3, 5]:
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
    for j in [3, 5, 1]:
        count = 0
        p = []
        for i in range(j):
            item = ranklist[i]
            if item in gtItem:
                count += 1
                p.append(count / (i + 1))
        if len(p) == 0:
            result.append(0)
        else:
            result.append(np.sum(p) / j)
    return result

def getRR(ranklist, gtItem):
    for i in range(5):
        item = ranklist[i]
        if item in gtItem:
            return 1/(i+1)
    return 0