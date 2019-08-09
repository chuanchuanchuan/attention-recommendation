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
import sys
import argparse
import multiprocessing as mp
from keras.engine.topology import Layer
import pickle
import math
import heapq  # for retrieval topK
import multiprocessing
from time import time
import matplotlib.pyplot as plt


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
    user_input = Input(shape=(44,),  name='user_input')
    item_input = Input(shape=(16,),  name='item_input')
    vector = concatenate([user_input, item_input])
    vector = Embedding(input_dim=53111, output_dim=100)(vector)
    print('0:', vector.shape)
    vector = Permute((2, 1))(vector)
    print('1:',vector.shape)
    vector = Reshape((4,1500))(vector)
    print('2:',vector.shape)
    vector = GlobalAveragePooling1D()(vector)
    print('3:', vector.shape)
    vector = Reshape((15, 100))(vector)
    print('4',vector.shape)
    O_seq = Attention(5, 120)([vector, vector, vector])
    print('5',O_seq.shape)
    O_seq = GlobalAveragePooling1D()(O_seq)
    print('6',O_seq.shape)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(1, activation='sigmoid')(O_seq)
    model = Model(inputs=[user_input, item_input],
                  outputs=outputs)
    return model



model = get_model(3056, 33138)
model.load_weights('music/Pretrain/savedmodel_0')
w = model.get_weights()



import pickle
import random

wd = 'music/'
#wd = ''
f = open(wd + 'user_info.txt', 'r')
user_info = f.readlines()
f.close()
f = open(wd + 'song_info.txt', 'r', encoding='utf8')
song_info = f.readlines()
f.close()

'''
读取音乐数据 
'''
song_info = song_info[1:]
song_dic = {}
for line in song_info:
    sname, sid, singer, album, cnum, zuoqu, zuoci, rateduser = line.split('\t')
    sid = int(sid)
    song_dic[sid] = [singer, album, zuoci, zuoqu]

'''
读取用户数据
'''
user_info = user_info[1:]
user_dic = {}
rated_song = set()

for line in user_info:
    uid, cnum, songls = line.split('\t')
    uid = int(uid)
    songls = eval(songls)
    songls = list(map(lambda x: int(x), songls))
    unknowsong = []
    for song in songls:
        if song not in song_dic:
            unknowsong.append(song)
        else:
            rated_song.add(song)
    for song in unknowsong:
        songls.remove(song)
    user_dic[uid] = list(set(songls))

uidls = list(user_dic.keys())
sidls = list(song_dic.keys())
print(len(rated_song))
'''
删除未出现过的歌曲
'''
for i in sidls:
    if i not in rated_song:
        sidls.remove(i)

'''
将用户、歌曲重新编号。使其为连续的自然数。
'''


def get_new_uid(uid):
    return uidls.index(uid)


def get_new_sid(sid):
    return sidls.index(sid)





# user_input=[14385, 14395, 14398, 14411, 14431, 14438, 14434, 14390, 14432, 14437, 47150]
#user_input = [46076, 46149, 38675, 46102, 30870, 0, 0, 0, 0, 0, 47178]
#user_input = [46076, 46149, 38675, 46102, 30870, 29217, 14985, 4765, 4796, 5472, 47178]

#user_input=[17986, 35784, 20161, 23189, 23419, 35903, 23303, 24167, 14371, 20417, 48057]


# item_input=[14385, 14378, 14379, 14378]
# item_input = [46149, 1027, 46132, 918]  Jay
#item_input=[46201, 1027, 46169, 2452]
#item_input=[17986, 17981, 17984, 17981]


with open(wd+"song_info_music_HK_w", "rb") as file:
    song_info = pickle.load(file)

with open(wd+"user_hist_withinfo_music_HK", "rb") as file:
    user_info = pickle.load(file)



def softmax2(c):
    newc = []
    for line in c:
        #line = line / np.max(np.abs(line))
        line=line/10
        #line = line / np.max(line)
        newc.append(np.exp(line) / sum(np.exp(line)))
    newc = np.array(newc)
    return newc


fc=[]
for uu in range(3000):
    if len(user_dic[uidls[uu]])<15:
        continue
    selected_user=uu
    song_raw=user_dic[uidls[selected_user]]
    #song_raw=random.sample(sidls,200)
    #song_raw = user_dic[uidls[1420]]
    song_new=list(map(get_new_sid,song_raw))
    cc=[]
    #song_new=[song_new[65]]
    for s in song_new:
        item_input=song_info[s]
        user_input=user_info[selected_user]
        vector = w[0][user_input + item_input]  # 60,100
        vector = vector.reshape((15, 4, 100))
        vector = vector.mean(axis=1)
        cl=[]
        for i in range(5):
            QQ = w[1]  # 100,600
            QQ = QQ.reshape(100, 5, 120)
            QQ = QQ[:, i, :]
            KK = w[2]
            KK = KK.reshape(100, 5, 120)
            KK = KK[:, i, :]
            VV = w[3]
            a = np.dot(vector, QQ)
            b = np.dot(vector, KK)
            c = np.dot(a, np.transpose(b))
            #for line in range(len(c)):
            #    c[line]=c[line]/np.max(np.abs(c[line]))
            cl.append(c)
        c=np.mean(cl,axis=0)
        cc.append(c)
    cc=np.array(cc)
    ncc = []
    for c in cc:
        ncc.append(softmax2(c))
    ncc = np.mean(ncc, axis=0)
    fc.append(ncc)


fc=np.array(fc)
ncc=fc
ncc=np.mean(ncc,axis=0)
np.save('ncc_avg.npy', ncc)

import seaborn as sns
sns.set()

ncc=ncc[:,[0,1,2,3,4,5,6,7,8,9,12,13,14]]
ncc=ncc[[0,1,2,3,4,5,6,7,8,9,12,13,14],]
for line in range(len(ncc)):
    ncc[line]=ncc[line]/np.sum(ncc[line])
ax2 = sns.heatmap(ncc,center=0.9)
plt.show()

exit()



import seaborn as sns

sns.set()

c = np.load('./music/cc.npy')

def softmax2(c):
    newc = []
    for line in c:
        #line = line / np.max(np.abs(line))
        line=line/5
        newc.append(np.exp(line) / sum(np.exp(line)))
    newc = np.array(newc)
    return newc

ax2 = sns.heatmap(softmax2(c),center=0.9)


cc = np.load('./music/cc.npy')
ncc=[]
for c in cc:
    ncc.append(softmax2(c))

ncc=np.mean(ncc,axis=0)
ncc=ncc[:,[0,1,2,3,4,5,6,7,8,9,12,13,14]]
ncc=ncc[[0,1,2,3,4,5,6,7,8,9,12,13,14],]
for line in range(len(ncc)):
    ncc[line]=ncc[line]/np.sum(ncc[line])


ncc = np.load('./music/ncc_avg.npy')
ncc=ncc[:,[0,1,2,3,4,5,6,7,8,9,12,13,14]]
ncc=ncc[[0,1,2,3,4,5,6,7,8,9,12,13,14],]
for line in range(len(ncc)):
    ncc[line]=ncc[line]/np.sum(ncc[line])
ax2 = sns.heatmap(ncc,center=0.9)
