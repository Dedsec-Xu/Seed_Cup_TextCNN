import re
import numpy as np

pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
source_file = './data/del50w/train_b.txt'
f = open(source_file)
data = f.read()
data = re.findall(pattern, data)[1:]

title_words = [data_item[1] for data_item in data]
describe_words = [data_item[3] for data_item in data]
words = []

for i in range(len(data)):
    if title_words[i] == '' and describe_words[i] =='':
        r=1
    else: 
        if title_words[i] == '':
            words += describe_words[i].split(',')
        else:
            if describe_words[i] =='': 
                words += title_words[i].split(',') 
            else:
                words += title_words[i].split(',') + describe_words[i].split(',')

wordsnum = []
for i in range(len(words)):
    wordsnum.append(int(words[i].strip('w')))

wordsnum.sort()
wordsfreq = []


wordsnum.sort()
wordsnum3=wordsnum
wordsnum3.append(0)
temp = 1
wordsfreq2 = []
for i in range(len(wordsnum)):
    if wordsnum3[i]==wordsnum3[i-1]:
        temp += 1
    else:
        wordsfreq2.append(temp)
        temp = 1

print('vocabsize is')
print(len(wordsfreq2))
