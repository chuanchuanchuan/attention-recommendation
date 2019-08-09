import numpy as np
l=[]
for i in range(60):
    l.append([np.floor(i/4)+1]*100)

l = np.array(l)
l.shape  # 60 100
l1=l.reshape(15,4,100).mean(axis=1)
l.shape




l=[]
for i in range(44):
    l.append([i % 4 + 1] * 100)

l = np.array(l)
l.shape
l.reshape(11,4,100).mean(axis=0)
l.shape