import re
import numpy as np
import matplotlib.pyplot as plt

HIGH_FREQ_GATE = 200000
HIGH_FREQ_RATE = 0.3

words_num = np.load('words.npy')
freq_unsorted = np.load('freq_unsorted.npy')
freq = np.load('freq.npy')
word_throwaway = []
for i, freqt in enumerate(freq_unsorted):
    if freqt > HIGH_FREQ_GATE:
        word_throwaway.append(i)

np.save('word_throwaway.npy',word_throwaway)
