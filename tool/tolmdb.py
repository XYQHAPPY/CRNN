import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import re
import Image
import numpy as np
import imghdr


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False		
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)
			
def createDataset(env, imagePathList, labelList, cnt, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    #env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    #cnt = 1
    for i in xrange(nSamples):   
        imagePath = ''.join(imagePathList[i]).split()[0]
        label = ''.join(labelList[i])
        print imagePath +' '+ label
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue	
		
        with open(imagePath, 'r') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 100 == 0:
            writeCache(env, cache)
            cache = {}
        print('Written %d / %d' % (cnt, nSamples))
        print cnt
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)
    return cnt
	

if __name__ == '__main__':
    outputPath = "/data/sata/share_sata/xyq/crnn/data/nlp_sentiment_val_lmdb"
    env = lmdb.open(outputPath, map_size=1099511627776)
    cnt = 0
    imgdata = open("/data/sata/share_sata/xyq/data/NLP/sentiment/label_test.txt")
    imagePathList = list(imgdata)
    labelList = []
    for line in imagePathList:
        word = line.split()[1]
        labelList.append(word)
    cnt = createDataset(env, imagePathList, labelList, cnt)
        #pass
