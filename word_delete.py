import re
import numpy as np
#delete every throwaway object

trainfpath = './data/train_final.txt'

trainfspath = './data/del50w/train_b.txt'
ta = np.load('word_throwaway.npy')
f1 = open(trainfpath, 'r')
f2 = open(trainfspath, 'w')

data = f1.read()
f1.close()
for i in ta:
    pattern1 = re.compile(',w'+str(ta)+'\t')
    pattern2 = re.compile(',w'+str(ta)+'\n')
    pattern3 = re.compile('w'+str(ta)+',')
    pattern4 = re.compile('\tw'+str(ta)+'\t')
    pattern5 = re.compile('\tw'+str(ta)+'\n')
    data = re.sub(pattern1, '\t', data)
    data = re.sub(pattern2, '\n', data)
    data = re.sub(pattern3, '', data)
    data = re.sub(pattern4, '\t\t', data)
    data = re.sub(pattern5, '\t\n', data)

f2.write(data)
f2.close()
