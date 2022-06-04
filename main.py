import json
import os.path
import csv

import jieba
from gensim import corpora, models, similarities
from gensim.test.utils import common_texts


class data_science:
    tfidf_model = None
    lsi_model = None
    w2VModel = None
    dic = None
    mSimilar = None
    bows = None
    text_list = None

    def __init__(self, is_load=True):
        if is_load is True:
            self.dic = corpora.Dictionary.load('Dic.dic')
            self.tfidf_model = models.TfidfModel.load('tfidf_model.tfidf')
            self.lsi_model = models.LsiModel.load("lsi_model.lsi")
            self.mSimilar = similarities.MatrixSimilarity.load('mSimilar.mSimilar')
            self.w2VModel = models.Word2Vec.load('w2V_model.w2v')
            self.bows = corpora.MmCorpus('bows.mm')
            self.text_list = []
            f = open("字段关联关系.csv", encoding="UTF-8")
            reader = csv.DictReader(f)
            for line in reader:
                self.text_list.append(jieba.lcut(line['数据元']))

    def data_from_ali(self):
        f = open("Ali.txt", encoding="UTF-8")
        text_list = []
        for line in f:
            text_list.append([json.loads(line)['sentence1'], json.loads(line)['sentence2']])

        dic = corpora.Dictionary(text_list)
        dic.save('Dic.dic')

        bows = [dic.doc2bow(text) for text in text_list]
        corpora.MmCorpus.serialize('bows.mm', bows)

        tfidf_model = models.TfidfModel(dictionary=dic)
        corpus_tfidf = tfidf_model[bows]
        tfidf_model.save('tfidf_model.tfidf')

        lsi_model = models.LsiModel(bows, id2word=dic, num_topics=150)
        corpus_lsi = lsi_model[bows]
        lsi_model.save('lsi_model.lsi')

        mSimilar = similarities.MatrixSimilarity(corpus_lsi)
        mSimilar.save('mSimilar.mSimilar')

        w2VModel = models.Word2Vec(sentences=text_list, vector_size=100, window=100000000, min_count=0, workers=4)
        w2VModel.save('w2V_model.w2v')


    def data_from_given(self):
        f = open("字段关联关系.csv", encoding="UTF-8")
        text_set = set()

        reader = csv.DictReader(f)
        for line in reader:
            # print(line['数据元'], line['字段中文（可忽略）'])
            text_set.add(line['数据元'])
            # print(line['数据元'])
            # text_list.append(jieba.lcut(line['字段中文（可忽略）']))

        text_list = []
        for i in text_set:
            text_list.append(jieba.lcut(i))



        dic = corpora.Dictionary(text_list)
        dic.save('Dic.dic')

        bows = [dic.doc2bow(text) for text in text_list]
        corpora.MmCorpus.serialize('bows.mm', bows)

        tfidf_model = models.TfidfModel(dictionary=dic)
        corpus_tfidf = tfidf_model[bows]
        tfidf_model.save('tfidf_model.tfidf')

        lsi_model = models.LsiModel(bows, id2word=dic, num_topics=50)
        corpus_lsi = lsi_model[bows]
        lsi_model.save('lsi_model.lsi')

        mSimilar = similarities.MatrixSimilarity(corpus_lsi)
        mSimilar.save('mSimilar.mSimilar')

        text_list_forv2W = []
        f.seek(0)
        reader = csv.DictReader(f)
        for line in reader:
            # print(line['数据元'], line['字段中文（可忽略）'])
            text_list_forv2W.append([line['数据元']])
            text_list_forv2W.append([line['字段中文（可忽略）']])

        # print(text_list_forv2W)

        w2VModel = models.Word2Vec(sentences=text_list_forv2W, window=5, min_count=1, negative=1, sample=0.01, workers=4)
        w2VModel.save('w2V_model.w2v')


    def word2Vec(self, s, topK):
        # print(self.w2VModel.wv.index_to_key)
        sims = self.w2VModel.wv.most_similar(s, topn=topK)
        # sims = self.w2VModel.wv.similar_by_word(s)
        for s in sims:
            print(s)

    def lsi(self, s, topK):
        vec_bow = self.dic.doc2bow(jieba.lcut(s))
        vec_tfidf = self.tfidf_model[vec_bow]
        vec_lsi = self.lsi_model[vec_tfidf]
        sims = self.mSimilar[vec_lsi]

        sims = sorted(enumerate(sims), key=lambda x: -x[1])
        for i in sims[:topK]:
            print(''.join(k for k in self.text_list[i[0]-1]), i[1])





    # model = Word2Vec(sentences=test_list, vector_size=100, window=5, min_count=1, workers=4)
    # model.save("word2vec.model")
    #
    #
    # # print(model.wv.index_to_key)
    # vector = model.wv["蚂蚁借呗等额还款可以换成先息后本吗"]
    # sims = model.wv.most_similar("蚂蚁借呗等额还款可以换成先息后本吗", topn=10)
    #
    # print(sims)


if __name__ == "__main__":
    sb = data_science(is_load=False)
    sb.data_from_given()

    dsb = data_science(is_load=True)
    dsb.lsi("开户人公民身份号码", 10)
    dsb.word2Vec("开户人公民身份号码", 10)

