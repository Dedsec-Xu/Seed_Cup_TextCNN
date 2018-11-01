import re
import numpy as np
import matplotlib.pyplot as plt

DOTS = 250

words_num = np.load('words.npy')
freq_unsorted = np.load('freq_unsorted.npy')
freq = np.load('freq.npy')
y1 = freq[len(freq)-DOTS:len(freq)]
x1 = []
for i in range(len(y1)):
    x1.append(i)

high_plot1 = plt.bar(x1,y1)
plt.show(high_plot1)

y2 = freq_unsorted[0:DOTS]
x2 = []
for i in range(len(y2)):
    x2.append(i)

high_plot2 = plt.bar(x2, y2)
plt.show(high_plot2)