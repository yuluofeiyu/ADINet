import os
from time import time
import time as T



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('ADINet train')
tic = time()
os.system('python train_ADINet.py')
hours, rem = divmod(time()-tic, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('ADINet test')
tic = time()
os.system('python test_ADINet.py')
hours, rem = divmod(time()-tic, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

#
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# print('ADINet evaluation')
# tic = time()
# os.system('python main.py')
# hours, rem = divmod(time()-tic, 3600)
# minutes, seconds = divmod(rem, 60)
# print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')