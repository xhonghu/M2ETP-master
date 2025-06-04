import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import slowfast_modules.slowfast as slowfast

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, load_pkl, slowfast_config, slowfast_args=None,
            use_bn=False, hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=1
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.conv2d = getattr(slowfast, c2d_type)(slowfast_config=slowfast_config,  slowfast_args=slowfast_args,
                                                  load_pkl=load_pkl, multi=True)
        self.conv1d = TemporalConv(input_size=1144,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.conv1d2 = TemporalConv(input_size=1408,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.conv1d3 = TemporalConv(input_size=2304,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)                             
        self.temporal_model1 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        self.temporal_model2 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        self.temporal_model3 = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        if weight_norm:
            self.classifier1 = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
            # self.classifier2 = NormLinear(hidden_size, self.num_classes)
            self.conv1d2.fc = NormLinear(hidden_size, self.num_classes)
            # self.classifier3 = NormLinear(hidden_size, self.num_classes)
            self.conv1d3.fc = NormLinear(hidden_size, self.num_classes)

        else:
            self.classifier1 = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
            # self.classifier2 = nn.Linear(hidden_size, self.num_classes)
            self.conv1d2.fc = nn.Linear(hidden_size, self.num_classes)
            # self.classifier3 = nn.Linear(hidden_size, self.num_classes)
            self.conv1d3.fc = nn.Linear(hidden_size, self.num_classes)

        if share_classifier:
            self.conv1d.fc = self.classifier1
            self.conv1d2.fc = self.classifier1
            self.conv1d3.fc = self.classifier1

    def forward(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            framewise1, framewise2, framewise3 = self.conv2d(x.permute(0,2,1,3,4))
        else:
            framewise = x

        # 数据和模型的列表
        framewise_data_list = [framewise1, framewise2, framewise3]
        conv1d_list = [self.conv1d, self.conv1d2, self.conv1d3]
        temporal_model_list = [self.temporal_model1, self.temporal_model2, self.temporal_model3]

        # 存储输出
        outputs = []
        con_output = []

        # 使用循环处理每个数据和模型
        for i in range(3):
            conv1d_outputs = conv1d_list[i](framewise_data_list[i], len_x)
            con_output.append(conv1d_outputs['conv_logits'])
            lgt = conv1d_outputs['feat_len']
            tm_outputs = temporal_model_list[i](conv1d_outputs['visual_feat'], lgt)
            outputs.append(self.classifier1(tm_outputs['predictions']))


        pred = None if self.training \
            else self.decoder.decode(outputs[2], lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(con_output[2], lgt, batch_first=False, probs=False)
        
        return {
            "feat_len": lgt,
            "conv_logits": con_output,
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                for i in range(3):
                    total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"][i].log_softmax(-1),
                                                        label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                        label_lgt.cpu().int()).mean()
                    loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                for i in range(3):
                    total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"][i].log_softmax(-1),
                                                        label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                        label_lgt.cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Dist':
                for i in range(3):
                    total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"][i],
                                                            ret_dict["sequence_logits"][i].detach(),
                                                            use_blank=False)
                    loss += total_loss['Dist']
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss