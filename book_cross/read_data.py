import numpy as np
import pickle
import pymysql
import random

random.seed(100)
workdir = ''

'''
连接数据库
for KSA
'''
conn = pymysql.connect(host='localhost',
                       port=3306,
                       user='root',
                       password='1234',
                       db='book_cross',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()

vocabulary = []
user_info = {}
book_info = {}
book_id_info = {}

'''
读取图书数据，将id和asin相关联
'''
cursor.execute("select ISBN, Book_Title, Book_Author, Year_Of_Publication, Publisher from BX_Books limit 50000")
results = cursor.fetchall()
row2newbookid = {}
index=0
for row in results:
    book_id = row['ISBN']

    if row['Book_Author'] == 'NA':
        author = 'Unknown'
    else:
        author = row['Book_Author']

    if row['Year_Of_Publication'] == 'NA':
        year = 'Unknown'
    else:
        year = row['Year_Of_Publication']

    if row['Publisher'] == 'NA':
        publisher = 'Unknown'
    else:
        publisher = row['Publisher']

    this_book_info = [book_id, author, year, publisher]
    #this_book_info = [author, year, publisher]
    t_book_info = []
    for i in this_book_info:
        if i not in vocabulary:
            vocabulary.append(i)
        t_book_info.append(vocabulary.index(i))

    new_id = index
    index+=1
    row2newbookid[book_id] = new_id
    '这个地方把图书的信息重复4次，是为了便于后续操作。'
    '一个用户在后面会表示为他交互过的3个物品在4个属性上的均值'
    '在bc_ksa的get_model函数里，首先把用户交互过的3个物品的共计12个属性的id，以及当前物品的4个属性的4次重复16个属性id作为embedding层输入'
    '然后得到的向量，每4个做一次平均，由此得到用户在四个属性上的向量均值，而物品属性保持不变（因为本来就重复了4次）'
    '''
    对应代码为
    user_input = Input(shape=(12,), name='user_input')
    item_input = Input(shape=(16,), name='item_input')
    vector = concatenate([user_input, item_input])
    vector = Embedding(input_dim=vocabulary_lenth + 1, output_dim=5)(vector)
    vector = Reshape((4, 35))(vector)
    vector = GlobalAveragePooling1D()(vector)
    '''
    book_info[new_id] = t_book_info*4
    book_id_info[new_id] = [t_book_info[0]]

'''
读取交互数据。只使用前50万条记录。
'''
cursor = conn.cursor()
cursor.execute("select User_ID, ISBN from Book_Ratings limit 500000")
results = cursor.fetchall()
pos = {}

index = 0
row2newuserid = {}
for row in results:
    row_user_id = int(row['User_ID'])
    if row_user_id not in row2newuserid:
        row2newuserid[row_user_id] = index
        index += 1
    new_user_id = row2newuserid[row_user_id]
    if new_user_id not in pos:
        pos[new_user_id] = []
    rowid = row['ISBN']
    if rowid not in row2newbookid:
        continue
    book_id = row2newbookid[rowid]
    pos[new_user_id].append(book_id)

index = 0
npos = {}
for i in range(len(pos)):
    if len(pos[i]) < 5:
        continue
    else:
        npos[index] = pos[i]
        index += 1

pos = npos

train = {}
if 'UNknown' not in vocabulary:
    vocabulary.append('UNknown')

f = open("bookcross.train.rating", "w")
h = open("bookcross.test.rating", "w")
g = open("bookcross.test.negative", "w")
unseencount = 0
unseenlist = []
for i in range(index):

    u_info = 'u' + str(i)
    vocabulary.append(u_info)
    #user_info[i] = [vocabulary.index(u_info)]
    user_info[i] = []
    all_pos = pos[i]
    random.shuffle(all_pos)
    test_pos = all_pos[0:5]
    if len(test_pos) != 5:
        print('Error')
    train_pos = all_pos[5:]

    for j in train_pos:

        for info in book_info[j][0:4]:
            user_info[i].append(info)
        #ISBN = book_info[j][0]
        #user_info[i].append(ISBN)
        if len(user_info[i]) >= 12:
            user_info[i]=user_info[i][0:12]
            break
    while len(user_info[i]) != 12:
        user_info[i].append(vocabulary.index(u_info))

    if len(train_pos) == 0:
        unseencount += 1
        unseenlist.append(i)
    train_neg = []
    test_neg = []
    while len(train_neg) < 4 * len(train_pos):
        n1 = random.randint(0, 49999)
        if n1 not in pos[i]:
            train_neg.append(n1)

    train[i] = train_neg + train_pos
    while len(test_neg) < 50:
        n1 = random.randint(0, 49999)
        if n1 not in train[i]:
            test_neg.append(n1)
    content = ''
    for j in range(len(train_pos)):
        content += str(i) + '\t' + str(train_pos[j]) + '\t' + '5' + '\n'

    for j in range(len(train_neg)):
        content += str(i) + '\t' + str(train_neg[j]) + '\t' + '0' + '\n'
    f.write(content)

    content = str(i)
    for j in range(len(test_pos)):
        content += '\t' + str(test_pos[j])
    content += '\n'
    h.write(content)

    content = '(' + str(i) + ',' + '1)'
    for n in test_neg:
        content += '\t' + str(n)
    content += '\n'
    g.write(content)

f.close()
h.close()
g.close()

pickle.dump(user_info, open('user_hist_withinfo_bookcross', 'wb'))
pickle.dump(book_info, open('book_info_bookcross', 'wb'))
pickle.dump(book_id_info, open('book_id_bookcross', 'wb'))

print(len(vocabulary))
print(unseencount)
'''
import pickle

wd='./amazon/'
with open(wd+"user_id_amazon", "rb") as file:
    user_info = pickle.load(file)

with open(wd+"book_info_amazon", "rb") as file:
    movie_info = pickle.load(file)



'''
