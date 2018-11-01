import re
import numpy as np
import matplotlib.pyplot as plt

source_file='./data/train_a.txt'
pattern = '.*?\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\t(.*?)\n'
f=open(source_file)
data = f.read()
data = re.findall(pattern, data)[1:]
title_words = [data_item[1] for data_item in data]
describe_words = [data_item[3] for data_item in data]
words = []
for i in range(len(data)):
    words += title_words[i].split(',')


words = set(words)
words = list(set(words))
words.sort()
words_num = []
for i, word in enumerate(words):
    words_num.append(int(words[i].strip('w')))


print(len(words_num))

words_num.sort()
np.save('words.npy',words_num)
freq = np.zeros(words_num[len(words_num)-1]+1)
for i, word in enumerate(words_num):
    freq[words_num[i]]=freq[words_num[i]]+1
words_absence = []
for i, freqt in enumerate(freq):
    if freqt == 0:
        words_absence.append(i)
freq_unsorted = freq
np.save('words_absence .npy',words_absence)
np.save('freq_unsorted.npy',freq_unsorted)
freq.sort()
np.save('freq.npy',freq)
