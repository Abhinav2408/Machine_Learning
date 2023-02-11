import math
import os
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()
trainaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part1_data/part1_data/train"
testaddress = "C:/Users/abhi2/Desktop/IIT DELHI COURSE/COL774/Assignment2/part1_data/part1_data/test"

if(len(sys.argv)!=1):
    trainaddress = sys.argv[1]
    testaddress = sys.argv[2]


postrainfiles = os.listdir(trainaddress + '/pos')
negtrainfiles = os.listdir(trainaddress + '/neg')


swords = stopwords.words('english')

#unigrams

wordfreq={}
poswordfreq={}
negwordfreq={}
poswords = []
negwords = []


#bigrams

biwordfreq = {}
biposwordfreq = {}
binegwordfreq = {}
biposwords = []
binegwords = []


#trigrams used as additional feature
triwordfreq = {}
triposwordfreq = {}
trinegwordfreq = {}
triposwords = []
trinegwords = []


for i in range(len(postrainfiles)):
    textfile = open(trainaddress + '/pos/'+postrainfiles[i], 'r',encoding='utf8')
    data = textfile.read()
    data = data.lower()
    data = re.sub('<br />', '', data)
    data = re.sub('\n', '', data)
    data = re.sub(r'[^\w\s]', '', data)
    textfile.close()
    z = [y for y in data.split() if y not in swords]
    x = [ps.stem(a) for a in z]
    bi = [x[i]+' '+x[i+1] for i in range(len(x)-1)]
    tri = [x[i]+' '+x[i+1]+' '+x[i+2] for i in range(len(x)-2)]
    poswords.extend(x)
    biposwords.extend(bi)
    triposwords.extend(tri)



for i in range(len(negtrainfiles)):
    textfile = open(trainaddress + '/neg/'+negtrainfiles[i], 'r',encoding='utf8')
    data = textfile.read()
    data = data.lower()
    data = re.sub('<br />', '', data)
    data = re.sub('\n', '', data)
    data = re.sub(r'[^\w\s]', '', data)
    textfile.close()
    z = [y for y in data.split() if y not in swords]
    x = [ps.stem(a) for a in z]
    bi = [x[i]+' '+x[i+1] for i in range(len(x)-1)]
    tri = [x[i]+' '+x[i+1]+' '+x[i+2] for i in range(len(x)-2)]
    negwords.extend(x)
    binegwords.extend(bi)
    trinegwords.extend(tri)





for word in poswords:
    if word in wordfreq:
        wordfreq[word] +=1
    else:
        wordfreq[word] = 1

    if word in poswordfreq:
        poswordfreq[word] += 1
    else:
        poswordfreq[word] = 1
    
for word in negwords:
    if word in wordfreq:
        wordfreq[word] +=1
    else:
        wordfreq[word] = 1

    if word in negwordfreq:
        negwordfreq[word] += 1
    else:
        negwordfreq[word] = 1






for word in biposwords:
    if word in biwordfreq:
        biwordfreq[word] +=1
    else:
        biwordfreq[word] = 1

    if word in biposwordfreq:
        biposwordfreq[word] += 1
    else:
        biposwordfreq[word] = 1
    
for word in binegwords:
    if word in biwordfreq:
        biwordfreq[word] +=1
    else:
        biwordfreq[word] = 1

    if word in binegwordfreq:
        binegwordfreq[word] += 1
    else:
        binegwordfreq[word] = 1








for word in triposwords:
    if word in triwordfreq:
        triwordfreq[word] +=1
    else:
        triwordfreq[word] = 1

    if word in triposwordfreq:
        triposwordfreq[word] += 1
    else:
        triposwordfreq[word] = 1
    
for word in trinegwords:
    if word in triwordfreq:
        triwordfreq[word] +=1
    else:
        triwordfreq[word] = 1

    if word in trinegwordfreq:
        trinegwordfreq[word] += 1
    else:
        trinegwordfreq[word] = 1

# print(len(wordfreq))
# print(len(poswordfreq))
# print(len(negwordfreq))

posvalues = sum(list(poswordfreq.values()))
negvalues = sum(list(negwordfreq.values()))

biposvalues = sum(list(biposwordfreq.values()))
binegvalues = sum(list(binegwordfreq.values()))

triposvalues = sum(list(triposwordfreq.values()))
trinegvalues = sum(list(trinegwordfreq.values()))




postheta = {}
negtheta = {}

bipostheta = {}
binegtheta = {}

tripostheta = {}
trinegtheta = {}
alpha = 1



for word in wordfreq.keys():
    if word in poswordfreq:
        postheta[word] = (poswordfreq[word] + alpha)/(posvalues + alpha*len(wordfreq))
    else:
        postheta[word] = alpha/(posvalues + alpha*len(wordfreq))

    if word in negwordfreq:
        negtheta[word] = (negwordfreq[word] + alpha)/(negvalues + alpha*len(wordfreq))
    else:
        negtheta[word] = alpha/(negvalues + alpha*len(wordfreq))



for word in biwordfreq.keys():
    if word in biposwordfreq:
        bipostheta[word] = (biposwordfreq[word] + alpha)/(biposvalues + alpha*len(biwordfreq))
    else:
        bipostheta[word] = alpha/(biposvalues + alpha*len(biwordfreq))

    if word in binegwordfreq:
        binegtheta[word] = (binegwordfreq[word] + alpha)/(binegvalues + alpha*len(biwordfreq))
    else:
        binegtheta[word] = alpha/(binegvalues + alpha*len(biwordfreq))



for word in triwordfreq.keys():
    if word in triposwordfreq:
        tripostheta[word] = (triposwordfreq[word] + alpha)/(triposvalues + alpha*len(triwordfreq))
    else:
        tripostheta[word] = alpha/(triposvalues + alpha*len(triwordfreq))

    if word in trinegwordfreq:
        trinegtheta[word] = (trinegwordfreq[word] + alpha)/(trinegvalues + alpha*len(triwordfreq))
    else:
        trinegtheta[word] = alpha/(trinegvalues + alpha*len(triwordfreq))


posphi = len(postrainfiles)/(len(postrainfiles) + len(negtrainfiles))
negphi = 1- posphi



#####    testing accuracy for using unigrams and bigrams   #####

postestfiles = os.listdir(testaddress+'/pos')
negtestfiles = os.listdir(testaddress + '/neg')

bicorrectcount = 0
biincorrectcount = 0

tricorrectcount = 0
triincorrectcount = 0

for filename in postestfiles:
    textfile = open(testaddress + '/pos/' + filename,'r', encoding='utf8')
    data = textfile.read()
    data = data.lower()
    data = re.sub('<br />', '', data)
    data = re.sub('\n', '', data)
    data = re.sub(r'[^\w\s]', '', data)
    textfile.close()
    z = [y for y in data.split() if y not in swords]
    x = [ps.stem(a) for a in z]
    bi = [x[i]+' '+x[i+1] for i in range(len(x)-1)]
    tri = [x[i]+' '+x[i+1]+' '+x[i+2] for i in range(len(x)-2)]

    biposprob = 0
    binegprob = 0

    triposprob = 0
    trinegprob = 0

    for word in x:
        if word in postheta:
            biposprob+=math.log(postheta[word])
            triposprob+=math.log(postheta[word])
        if word in negtheta:
            binegprob+=math.log(negtheta[word])
            trinegprob+=math.log(negtheta[word])

    for word in bi:
        if word in bipostheta:
            biposprob+=math.log(bipostheta[word])
        if word in binegtheta:
            binegprob+=math.log(binegtheta[word])

    
    for word in tri:
        if word in tripostheta:
            triposprob+=math.log(tripostheta[word])
        if word in trinegtheta:
            trinegprob+=math.log(trinegtheta[word])

    biposprob+=math.log(posphi)
    binegprob+=math.log(negphi)

    triposprob+=math.log(posphi)
    trinegprob+=math.log(negphi)

    if(biposprob>binegprob):
        bicorrectcount+=1
    else:
        biincorrectcount+=1

    if(triposprob>trinegprob):
        tricorrectcount+=1
    else:
        triincorrectcount+=1



for filename in negtestfiles:
    textfile = open(testaddress + '/neg/' + filename,'r', encoding='utf8')
    data = textfile.read()
    data = data.lower()
    data = re.sub('<br />', '', data)
    data = re.sub('\n', '', data)
    data = re.sub(r'[^\w\s]', '', data)
    textfile.close()
    z = [y for y in data.split() if y not in swords]
    x = [ps.stem(a) for a in z]
    bi = [x[i]+' '+x[i+1] for i in range(len(x)-1)]
    tri = [x[i]+' '+x[i+1]+' '+x[i+2] for i in range(len(x)-2)]

    biposprob = 0
    binegprob = 0

    triposprob = 0
    trinegprob = 0

    for word in x:
        if word in postheta:
            biposprob+=math.log(postheta[word])
            triposprob+=math.log(postheta[word])
        if word in negtheta:
            binegprob+=math.log(negtheta[word])
            trinegprob+=math.log(negtheta[word])

    for word in bi:
        if word in bipostheta:
            biposprob+=math.log(bipostheta[word])
        if word in binegtheta:
            binegprob+=math.log(binegtheta[word])

    
    for word in tri:
        if word in tripostheta:
            triposprob+=math.log(tripostheta[word])
        if word in trinegtheta:
            trinegprob+=math.log(trinegtheta[word])

    biposprob+=math.log(posphi)
    binegprob+=math.log(negphi)

    triposprob+=math.log(posphi)
    trinegprob+=math.log(negphi)

    if(biposprob<binegprob):
        bicorrectcount+=1
    else:
        biincorrectcount+=1

    if(triposprob<trinegprob):
        tricorrectcount+=1
    else:
        triincorrectcount+=1

biaccuracy = (100*bicorrectcount)/(bicorrectcount+biincorrectcount)
triaccuracy = (100*tricorrectcount)/(tricorrectcount+triincorrectcount)

print("Testing Accuracy along with unigrams and bigrams is",biaccuracy,"%")
print("Testing Accuracy along with unigrams and trigrams(additonal feature) is",triaccuracy,"%")

