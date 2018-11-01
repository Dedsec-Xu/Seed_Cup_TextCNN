import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F
torch.manual_seed(1)

def kmax_pooling(x, dim, k):
     index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
     return x.gather(dim, index)

class FC(nn.Module):
    def __init__(self, opt):
        super(FC, self).__init__()
        #self.kernel_sizes = kwargs.get('kernel_sizes', [1, 3, 5])
        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
        self.NUM_ID_FEATURE_MAP = opt.NUM_ID_FEATURE_MAP
        self.USE_CUDA = opt.USE_CUDA
        self.opt=opt
        self.word_lstm = nn.LSTM(input_size = opt.EMBEDDING_DIM,\
                            hidden_size = opt.HIDDEN_SIZE,
                            num_layers = opt.NUM_LAYERS,

                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )
        # question_convs1 = [nn.Sequential(
        #         nn.Conv1d(in_channels=opt.EMBEDDING_DIM,
        #                   out_channels=opt.TITLE_DIM,
        #                   kernel_size=kernel_size),
        #         nn.BatchNorm1d(opt.TITLE_DIM),
        #         nn.ReLU(inplace=True),

        #         nn.MaxPool1d(kernel_size=(opt.SENT_LEN - kernel_size + 1))
        #     )for kernel_size in opt.SIN_KER_SIZE]

        self.question_convs = nn.Sequential(
                nn.Conv1d(in_channels=opt.HIDDEN_SIZE*2+opt.EMBEDDING_DIM,
                          out_channels=opt.TITLE_DIM,
                          kernel_size=3),
                nn.BatchNorm1d(opt.TITLE_DIM),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=opt.TITLE_DIM,
                          out_channels=opt.TITLE_DIM,
                          kernel_size=3),
                nn.BatchNorm1d(opt.TITLE_DIM),
                nn.ReLU(inplace=True),
                #nn.MaxPool1d(kernel_size=(opt.SENT_LEN - kernel_size[0] - kernel_size[1] + 2))
            )#for kernel_size in opt.DOU_KER_SIZE]

        #question_convs = question_convs1
        #question_convs.extend(question_convs2)

        self.num_seq = 1
        self.change_dim_conv  = nn.Conv1d(opt.TITLE_DIM*self.num_seq, opt.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)
        self.standard_pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        self.standard_batchnm = nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP)
        self.standard_act_fun = nn.ReLU()

        #self.question_convs = nn.ModuleList(question_convs)
        self.fc1 = nn.Sequential(
            nn.Linear(opt.NUM_ID_FEATURE_MAP*2, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(opt.NUM_ID_FEATURE_MAP*2, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(opt.NUM_ID_FEATURE_MAP*2, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASS_3)
        )

    def forward(self, question):
        question = self.encoder(question)
        question_out = self.word_lstm(question.permute(1,0,2))[0].permute(1,2,0) 
        question_em = question.permute(0,2,1)
        question1 = torch.cat((question_out,question_em),dim=1)

        x = kmax_pooling(self.question_convs(question1),2,self.opt.kmax_pooling)
        #x  = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        #x  = torch.cat(x, dim=1)
        xp = x
        xp = self.change_dim_conv(xp)
        x  = self.conv3x3(in_channels=x.size(1), out_channels=self.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = self.conv3x3(self.NUM_ID_FEATURE_MAP, self.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = x+xp
        while x.size(2) > 2:
            x = self._block(x)
        x  = x.view(x.size(0), -1)
        #print('1')
        output_1  = self.fc1(x)
        output_2  = self.fc2(x)
        output_3  = self.fc3(x)
        return (output_1, output_2, output_3)

    def conv3x3(self, in_channels, out_channels, stride=1, padding=1):
        """3x3 convolution with padding"""
        _conv =  nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                         padding=padding, bias=False)
        if self.USE_CUDA:
            return _conv.cuda()
        else:
            return _conv

    def _block(self, x):
        x  = self.standard_pooling(x)
        xp = x
        x  = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x += xp
        return x