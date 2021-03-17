import gensim
import numpy as np
import linecache

def load_glove_embedding(word_list,uniform_scale,dimension_size):
    glove_words = {}
    file_path = 'data/glove.6B.50d_test.txt'
    with open(file_path,'r',encoding='UTF-8') as fopen:
        for line in fopen:
            tokens = line.rstrip().split()
            glove_words[tokens[0]] = np.array(tokens[1:],dtype='float32')
    word_vectors = []
    for word in word_list:
        if word in glove_words:
            word_vectors.append(glove_words[word])
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size,dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale,uniform_scale,dimension_size))
    return word_vectors


# def load_glove_embedding(word_list,uniform_scale,dimension_size):
#     glove_words = []
#     file_path = 'data/glove.840B.300d.txt'
#     with open(file_path,'r',encoding='UTF-8') as fopen:
#         for line in fopen:
#             glove_words.append(line.strip())
#     word2offset = {w:i for i,w in enumerate(glove_words)}
#     word_vectors = []
#     for word in word_list:
#         if word in word2offset:
#             line = linecache.getline(file_path,word2offset[word]+1)
#             assert(word==line[:line.find(' ')].strip())
#             word_vectors.append(np.fromstring(line[line.find(' '):].strip(),sep=' ',dtype=np.float32))
#         elif word == '<pad>':
#             word_vectors.append(np.zeros(dimension_size,dtype=np.float32))
#         else:
#             word_vectors.append(np.random.uniform(-uniform_scale,uniform_scale,dimension_size))
#     return word_vectors   

def load_aspect_embedding_from_w2v(aspect_list,word_stoi,w2v):
    aspect_vectors = []
    for w in aspect_list:
        aspect_vectors.append(w2v[word_stoi[w.split()[0]]])
    return aspect_vectors


# def load_aspect_embedding_from_file(aspect_list,file_path):
#     aspect_vectors = {}
#     d = 0
#     with open(file_path,'r',encoding='UTF-8') as fopen:
#         for line in fopen:
#             w,v = line.split(':')
#             v = np.fromstring(v,sep=' ')
#             d = len(v)
#             aspect_vectors[w] = v
#     vecs = []
#     for a in aspect_list:
#         if a not in aspect_vectors:
#             vecs.append(np.random.uniform(-0.25,0.25,d))
#         else:
#             vecs.append(aspect_vectors[a.lower()])
#     return vecs,d


