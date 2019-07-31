#coding: utf-8
import torch
from torch.autograd import Variable
import utils
import dataset
import os
import keys
from PIL import Image
import numpy as np
import models.crnn as crnn

#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
model_path = './mymodels_nlp_sentiment/netCRNN_152_25.pth'
#model_path = './data/netCRNN63.pth'
img_path = '/data/sata/share_sata/xyq/NLP/mysentiment/test.png'
#alphabet = u'\'ACIMRey万下依口哺摄次状璐癌草血运重'
#alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = keys.alphabet
#print(alphabet)
nclass = len(alphabet) + 1
model = crnn.CRNN(32, 1, nclass, 128).cuda()#使用GPU
#model = crnn.CRNN(32, 1, nclass, 256).cpu()#使用CPU
#model = crnn.CRNN(32, 1, nclass, 256,1).cuda()

print('loading pretrained model from %s' % model_path)
pre_model = torch.load(model_path)#使用GPU
#pre_model = torch.load(model_path,map_location='cpu')#使用CPU
for k,v in pre_model.items():
    print(k,len(v))
#model.load_state_dict(pre_model)
model_dict = model.state_dict()
for k,v in model_dict.items():
    model_dict[k] = pre_model['module.'+k]            			
model.load_state_dict(model_dict)

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image).cuda()#使用GPU
#image = transformer(image).cpu()#使用CPU
image = image.view(1, *image.size())
image = Variable(image)

tempeval = model.eval()
preds = model(image)

#tempkk = np.argsort(-preds.data,axis=2)
#print tempkk[:,:,0:5]
predstemp, preds = preds.max(2)
#preds = preds.squeeze(2)
preds = preds.squeeze(1)
#preds = preds.transpose(1, 0).contiguous().view(-1)
preds = preds.transpose(-1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred.encode('utf8'), sim_pred.encode('utf8')))
