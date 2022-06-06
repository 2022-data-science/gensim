# -*- coding: utf-8 -*-

import logging

from gensim.models import word2vec
from gensim import models

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("111.txt")
    model = word2vec.Word2Vec(sentences, vector_size=250, workers=16, min_count=0)

    #保存模型，供日後使用
    model.save("word2vec.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
    model = models.Word2Vec.load('word2vec.model')

    res = model.wv.most_similar("开采矿种", topn=100)
    for item in res:
        print(item[0] + "," + str(item[1]))

    # print(model.wv)

    print(model.wv.similarity("开采矿种", "开采主矿种"))
