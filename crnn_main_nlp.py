#coding: utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import chardet
import keys
import sys 
import collections
import pandas as pd
from sklearn.model_selection import train_test_split
from pymodels import *
reload(sys)  
sys.setdefaultencoding('utf8')

import models.crnn_nlp as crnn_nlp

os.environ["CUDA_VISIBLE_DEVICES"] ="1"
str1 = keys.nlp_key
parser = argparse.ArgumentParser()
parser.add_argument('--trainroot',  type=str,default='./data/fjlaxx_train_lmdb', help='path to dataset')
parser.add_argument('--valroot',  type=str, default='./data/fjlaxx_test_lmdb',help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=1000, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True,help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default= str1)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--experiment', type=str,default='./mymodels_nlp/', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=30, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=20000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', default=True,help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


readdata = open('output.txt','r').readlines()
data = []
for ind, item in enumerate(readdata):
    y = item.split('|')[0]
    x = "|".join(item.split('|')[1:])
    data.append([y,x])
# unique_class = pd.unique(y)
count_y = pd.value_counts(np.array(data)[:,0])
count_y_df = pd.DataFrame(count_y)
drop_class = count_y_df[count_y_df[0]>150]
good_class = drop_class.index.values.tolist()
alldata = pd.DataFrame(data=data,columns=['label','data_V'])
traintestdata = alldata[alldata['label'].isin(good_class)]
X_train, X_test, y_train, y_test = train_test_split(traintestdata['data_V'].values.tolist()
                                                    , traintestdata['label'].values.tolist()
                                                    , test_size=0.2, random_state=2019)

wordlist = open('Tencent_AILab_Chinese.txt','r').read()
wordlist = eval(wordlist)
embeddinglist = open('Tencent_AILab_ChineseEmbedding.txt','r').readlines()[1:]

if '|' in wordlist:
    print('ok')
train_dataset = dataset.TextDataset(X_train,y_train,good_class,wordlist,embeddinglist)
test_dataset = dataset.TextDataset(X_test,y_test,good_class,wordlist,embeddinglist)

if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=opt.batchSize,
#     shuffle=True, sampler=sampler,
#     num_workers=int(opt.workers),
#     collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
# test_dataset = dataset.lmdbDataset(
#     root=opt.valroot, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers))

alphabet = opt.alphabet.decode('utf-8')

nclass = len(alphabet) + 1
nc = 1

criterion = nn.CrossEntropyLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# crnn = crnn_nlp.CRNN_nlp(opt.imgH, nc, nclass, opt.nh)
model = resnet101()
model.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    pre_trainmodel = torch.load(opt.crnn)
    model_dict = crnn.state_dict()
    weig1 = 'rnn.1.embedding.weight'
    bias1 = 'rnn.1.embedding.bias'
    #if len(model_dict[weig1]) == len(pre_trainmodel[weig1]) and len(model_dict[bias1]) == len(pre_trainmodel[bias1]):
    #    crnn.load_state_dict(pre_trainmodel)
    #else :
    for k, v in model_dict.items():
        print(k)
    for k,v in model_dict.items():
        pre = pre_trainmodel['module.'+k]
        mo = model_dict[k]
        print(pre.shape,mo.shape)
        if pre.shape == mo.shape:
            model_dict[k] = pre_trainmodel['module.'+k]
    crnn.load_state_dict(model_dict)
print(model)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)


def val(net, test_dataset, criterion, max_iter=2000):
    print('Start val')

    for p in model.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    loss_all = 0.0
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_C, cpu_y = data

        preds = model(cpu_C.cuda().float())
        cost = criterion(preds, cpu_y.cuda())

        loss_all += cost
        _, preds = preds.max(1)
        for pred, target in zip(preds, cpu_y.cuda()):
            if pred == target:
                n_correct += 1
        if i % 500 == 0:
            print(str(i) + '/' + str(max_iter) + ',acc=' + str(n_correct / 1.0 / i / opt.batchSize))
        if i >= 1000:
            break

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_all / max_iter, accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_X, cpu_y = data

    preds = model(cpu_X.cuda().float())
    cost = criterion(preds, cpu_y.cuda())
    model.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    while i < len(train_loader):
        cost = trainBatch(model, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % 10 == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % 10000 == 0:
            val(model, test_dataset, criterion)
            for p in model.parameters():
                p.requires_grad = True
            model.train()

        # do checkpointing
        if i % 10000 == 0:
            torch.save(
                model.state_dict(), '{0}/net_{1}_{2}.pth'.format(opt.experiment, epoch, i))
