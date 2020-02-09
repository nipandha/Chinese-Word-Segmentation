import numpy as np
from scipy.sparse import csr_matrix
import random
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math
encoding = 'big5hkscs'
lines = []
num_errors = 0
num_lines = 0
for line in open('training.txt', 'rb'):
    try:
        lines.append(line.decode(encoding))
        num_lines += 1
    except UnicodeDecodeError as e:
        num_errors += 1
print('Encountered %d decoding errors.' % num_errors)
# The `lines` list contains strings you can use.
char_map = {}
char_map['३']=1 
char_map['४']=2
bigram_map = {}
char_index = 3
bigram_index = 1

def isHkscsOrPunct(c):
    if (ord(c)>=19968 and ord(c)<=40959):
        return True
    return False

for raw_line in lines:
    words =  raw_line.rstrip().split() 
    line = '३' + ''.join(words) + '४'
    for ci in range(len(line)):
        if line[ci] != " ":
            if line[ci] not in char_map:
                char_map[line[ci]] = char_index
                char_index +=1
            if ci+1<len(line) and line[ci+1]!=" ":
                b = line[ci : ci+2]
                if b not in bigram_map:
                    bigram_map[b] = bigram_index
                    bigram_index +=1
def getVectors(line):
    l = '३' + line.rstrip() + '४'
    vectors_x = []
    vectors_y = []
    i = 0
    n = len(l) - 5 + 1
    for i in range(n):
        if(l[i]==" "):
            continue
        j = i
        char = 0
        abcd = ["0","0","0","0"]
        label = 0
        while j < len(l):
            if char<4:
                if l[j] != " ":
                    abcd[char] = l[j]
                    j +=1
                    char +=1
                else:
                    j +=1
                    if char == 2:
                        label = 1
            if char == 4:
                ab = ''.join(abcd[0 : 2])
                b = abcd[1]
                bc = ''.join(abcd[1 : 3])
                c = abcd[2]
                cd = ''.join(abcd[2 : 4])
                v = [ bigram_map[ab], char_map[b], bigram_map[bc], char_map[c] , bigram_map[cd] ]
                #v = [ ab, b, bc, c , cd ]
                vectors_x.append(v)
                vectors_y.append(label)
                break
    return vectors_x, vectors_y
lines = lines[0:90000]
random.shuffle(lines)
N = len(lines)
train_N = math.ceil(0.8 * N)
train = lines[0:train_N]
validation = lines[train_N:N]
def getBatches(dataset, batchSize = 40):
    while True:
        random.shuffle(dataset)
        batchesX = []
        batchesY = []
        for i in range(0,len(dataset),batchSize):
            batch = dataset[i : i + batchSize]
            current_batch_sz = len(batch)
            X = []
            Y = []
            for line in batch:
                if len(line)<50:
                    x, y = getVectors(line)
                    X.append(x)
                    Y.append(y)
            maxlen = len(max(X, key=len))
            input = np.zeros((current_batch_sz, maxlen, 5))
            labels = np.zeros((current_batch_sz, maxlen))
            for j in range(len(X)):
                seq = X[j]
                for k in range(len(seq)):
                    input[j][k] = seq[k]
                    labels[j][k] = Y[j][k]
            yield input, labels
def getData(dataset):
    random.shuffle(dataset)
    X = []
    Y = []
    n = len(dataset)
    for line in dataset:
        if len(line) <50:
            x, y = getVectors(line)
            X.append(x)
            Y.append(y)
    maxlen = len(max(X, key=len))
    input = np.zeros((n, maxlen, 5))
    labels = np.zeros((n, maxlen))
    for j in range(len(X)):
        seq = X[j]
        for k in range(len(seq)):
            input[j][k] = seq[k]
            labels[j][k] = Y[j][k]
    return input,labels
class BiLstm(tf.keras.Model):
  def __init__(self, charset_size, bigramset_size, emb_dim_char, emb_dim_bigram, units, batch_sz, dropout = 0.25):
    super(BiLstm, self).__init__()
    self.batch_sz = batch_sz
    self.units = units
    self.embedding_char = tf.keras.layers.Embedding(charset_size, emb_dim_char)
    self.embedding_bigram = tf.keras.layers.Embedding(bigramset_size, emb_dim_bigram)
    self.forward_LSTM = tf.keras.layers.LSTM(self.units,
                                   dropout = dropout, 
                                   return_sequences=True,
                                   return_state=True)
    self.BiLstmLayer = tf.keras.layers.Bidirectional(self.forward_LSTM)
    self.concatenate = tf.keras.layers.Concatenate(axis = 2)
    self.fc = tf.keras.layers.Dense(1,activation='sigmoid')

  def call(self, inputs):
    l = tf.unstack(inputs,axis=2)
    x1 = self.embedding_bigram(l[0])
    x2 = self.embedding_char(l[1])
    x3 = self.embedding_bigram(l[2])
    x4 = self.embedding_char(l[3])
    x5 = self.embedding_bigram(l[4])
    embedding = self.concatenate([x1,x2,x3,x4,x5])
    output, state, *_ = self.BiLstmLayer(embedding)
    logits = tf.squeeze(self.fc(output), axis=2)
    return logits
model = BiLstm(charset_size = len(char_map)+3, bigramset_size=len(bigram_map)+1, emb_dim_char = 40,emb_dim_bigram =70,
              units = 150, batch_sz = 120)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath="weights-improvement{epoch}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
callbacks_list = [checkpoint,early]
train_dataset = getBatches(train)
vx,vy = getData(validation)
model.fit(train_dataset, validation_data = (vx,vy), steps_per_epoch = train_N/40,
    epochs=200, callbacks=callbacks_list, verbose=1)
