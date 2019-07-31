#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import word2vec
import logging
import jieba
from PIL import Image
import numpy as np
 
 
# 主程序
# logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
# sentences =word2vec.Text8Corpus(u"fenci_result.txt")  # 加载语料
# model =word2vec.Word2Vec(sentences, size=256,min_count=1)  #训练skip-gram模型，默认window=5
#
# print model
# # 计算两个词的相似度/相关程度
# try:
#     y1 = model.similarity(u"经济", u"业务")
# except KeyError:
#     y1 = 0
# print u"【经济】和【业务】的相似度为：", y1
# print"-----\n"
# #
# # 计算某个词的相关词列表
# y2 = model.most_similar(u"风险", topn=20)  # 20个最相关的
# print u"和【风险】最相关的词有：\n"
# for item in y2:
#     print item[0], item[1]
# print"-----\n"
#
# # 寻找对应关系
# print u"中国经济网"
# y3 =model.most_similar([u'中国', u'经济'], [u'网'], topn=3)
# for item in y3:
#     print item[0], item[1]
# print"----\n"
#
# # 寻找不合群的词
# y4 =model.doesnt_match(u"金融服务 馆 由 福建省 金融".split())
# print u"不合群的词：", y4
# print"-----\n"
#
# # 保存模型，以便重用
#model.save(u"skycent.model")
# 对应的加载方式
def toTrainAndTest():
    file1 = open('/data/sata/share_sata/xyq/data/NLP/sentiment/sentenceAndLabel.txt')
    sentences = file1.readlines()
    poscount = 0
    negcount = 0
    posandnegcount = 0
    for item in sentences:
        file2 = open(fileroot + "label_train.txt", 'a')
        file3 = open(fileroot + "label_test.txt", 'a')
        # sentence = u'最近，微博大号“王大力如山”爆料称，有类似“金融小伙伴”的培训机构公然声称推出保证拿到金融机构实习offer的收费内推的“服务项目”，涉及机构囊括了券商、基金、银行、互金等多个金融机构。消息甫一出现就引起轩然大波，并持续发酵。'
        sentence = item.replace('\n', '').split(' ')[0]
        label = item.replace('\n', '').split(' ')[1]
        seg_list = jieba.cut(sentence, cut_all=False)
        sentence_list = " ".join(seg_list).split()
        vecs = []
        vecs.append((model_2[[itemword for itemword in sentence_list]] - kana.min()) * rait)
        im = Image.fromarray(np.uint8(vecs).squeeze(0))
        if label == '1':
            label = '正'
            poscount += 1
            if poscount % 6 == 0:
                im.save(fileroot + 'test/' + str(poscount) + '_' + label + '.png')
                file3.write(fileroot + 'test/' + str(poscount) + '_' + label + '.png' + ' ' + label + '\n')
            else:
                im.save(fileroot + 'train/' + str(poscount) + '_' + label + '.png')
                file2.write(fileroot + 'train/' + str(poscount) + '_' + label + '.png' + ' ' + label + '\n')
        if label == '-1':
            label = '敏'
            negcount += 1
            if negcount % 6 == 0:
                im.save(fileroot + 'test/' + str(negcount) + '_' + label + '.png')
                file3.write(fileroot + 'test/' + str(negcount) + '_' + label + '.png' + ' ' + label + '\n')
            else:
                im.save(fileroot + 'train/' + str(negcount) + '_' + label + '.png')
                file2.write(fileroot + 'train/' + str(negcount) + '_' + label + '.png' + ' ' + label + '\n')
        if label == '0':
            label = '客'
            posandnegcount += 1
            if posandnegcount % 6 == 0:
                im.save(fileroot + 'test/' + str(posandnegcount) + '_' + label + '.png')
                file3.write(fileroot + 'test/' + str(posandnegcount) + '_' + label + '.png' + ' ' + label + '\n')
            else:
                im.save(fileroot + 'train/' + str(posandnegcount) + '_' + label + '.png')
                file2.write(fileroot + 'train/' + str(posandnegcount) + '_' + label + '.png' + ' ' + label + '\n')
        file2.close()
        file3.close()

    file1.close()
def toimg(sentence):
    sentence = sentence.replace('\n', '').replace(' ','').replace('\t','')
    seg_list = jieba.cut(sentence, cut_all=False)
    sentence_list = " ".join(seg_list).split()
    vecs = []
    vecs.append((model_2[[itemword for itemword in sentence_list if itemword in model_2.wv.index2word]] - kana.min()) * rait)
    im = Image.fromarray(np.uint8(vecs).squeeze(0))
    im.save('/data/sata/share_sata/xyq/NLP/mysentiment/test.png')

fileroot = '/data/sata/share_sata/xyq/data/NLP/sentiment/'
model_2 =word2vec.Word2Vec.load("skycent.model")
kana = model_2.wv.vectors
rait = 256/(kana.max()-kana.min())
toimg(u'中国经济网编者按：2017年2月20日，福建海峡环保集团股份有限公司（以下称“海峡环保”，股票代码603817）在上交所挂牌上市，保荐机构为兴业证券，发行数量为11,250万股，全部为新股发行，无老股转让。')
toTrainAndTest()
#for item in kana:
#    print item.max()
#print kana.max()
# 以一种c语言可以解析的形式存储词向量
#model.save_word2vec_format(u"书评.model.bin", binary=True)
# 对应的加载方式
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)
