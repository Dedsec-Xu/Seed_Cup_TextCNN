import torch
class Config(object):
    def __init__(self):
        self.USE_CUDA           =       torch.cuda.is_available()
        self.NUM_EPOCHS         =       100
        self.TRAIN_BATCH_SIZE   =       128
        self.VAL_BATCH_SIZE     =       128
        self.TEST_BATCH_SIZE    =       128
        self.MODEL_FILE         =       './model.t7'
        
        self.TRAIN_FILE         =       './data/train_b.txt'
        self.VAL_FILE           =       './data/valid_b.txt'
        self.TEST_FILE          =       './data/test_b.txt'
        self.ANS_FILE           =       './data/ans.txt'
        self.LR                 =       1e-3

        self.NUM_CLASS_1        =       20
        self.NUM_CLASS_2        =       135
        self.NUM_CLASS_3        =       265
        self.EMBEDDING_DIM      =       300
        self.VOCAB_SIZE         =       179482
        self.KERNEL_SIZE        =       [2, 3, 4, 5]

        self.TITLE_DIM          =       512
        self.LINER_HID_SIZE     =       1024
        self.SENT_LEN           =       100
        self.HIDDEN_SIZE        =       256
        self.NUM_LAYERS         =       2
        self.kmax_pooling       =       2

        # TextCNNInc模型设置
        self.SIN_KER_SIZE = [1, 3]  # single convolution kernel
        self.DOU_KER_SIZE = [(1, 3), (3, 5)]  # double convolution kernel

        # TextCNNIncDeep模型设置
        self.NUM_ID_FEATURE_MAP = 250
    def get_lr(self,epoch):
        if (epoch+1)%10==0 and self.LR>1e-7:
            self.LR*=0.1
        print("learning rate:",self.LR)
        return self.LR
