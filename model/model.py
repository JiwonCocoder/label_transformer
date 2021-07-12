import math
from torch.nn import functional as F
from .backbone import *
import pdb

class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim//num_heads
        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt)) #(128, 32)
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2*self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt*num_heads, fdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)
        d = math.sqrt(self.fatt)

        Nx = len(fx_in)
        f = torch.cat([fx_in, fp_in])
        pdb.set_trace()
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        fx, fp = f[:, :Nx], f[:, Nx:]

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w
class AttenHeadX(nn.Module):
    def __init__(self, fdim, num_heads=8, num_classes=10):
        super().__init__()
        self.nhead = num_heads

        # self.intermediateDim = fdim + num_classes #(128 + 10)
        self.num_layers = 6
        #self.intermediateDim, self.num_layers
        # is based on Attention is All you Need
        self.embFC = nn.Linear(fdim, fdim - num_classes)
        #(in) 128, (out) 512 - 10
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = fdim, nhead = self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = self.num_layers)
        #Encoder : in(1, S, 138), out(1, S, 138)
        # self.deEmbFC = nn.Linear(self.intermediateDim, fdim)
        '''
        TransformerEncoder is a stack of N encoder layers
        Args:
            encoder_layer: an instance of the TransformerEncoderLayer () class (required)
            num_layers: the number of sub-encoder-layer in the encoder(required)
            norm: the layer normalization component (optional)
        '''

    def forward(self, fx, cls_xf):
        proj_fx = self.embFC(fx) #((bl+bu)*k, 128-num_classes)
        fx_added_cls = torch.cat([proj_fx, cls_xf], dim=1) #((bl+bu)*k, 128)
        fx_added_cls = fx_added_cls.unsqueeze(0) #(1, (bl+bu)*k, 128)
        gx = self.transformer_encoder(fx_added_cls)
        return gx

class FeatMatch(nn.Module):
    def __init__(self, backbone, num_classes, devices, num_heads=1, amp=True, attention='Feat'):
        super().__init__()
        self.mode = 'train'
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.devices = devices
        self.default_device = torch.device('cuda', devices[0]) if devices is not None else torch.device('cpu')
        fext, self.fdim = make_backbone(backbone)
        print(self.devices)
        self.fext = nn.DataParallel(AmpModel(fext, amp), devices)
        print(attention)
        if attention == 'Feat':
            print("soft attention")
            self.atten = AttenHead(self.fdim, num_heads)
        elif attention == 'Transformer':
            print("transformer")
            self.atten = AttenHeadX(self.fdim, num_heads, num_classes)
        print(self.atten)
        self.clf = nn.Linear(self.fdim, num_classes)

    def set_mode(self, mode):
        self.mode = mode

    def extract_feature(self, x):
        return self.fext(x)

    def forward(self, x, fp=None):
        if self.mode == 'fext':
            return self.extract_feature(x)

        elif self.mode == 'pretrain':
            fx = self.extract_feature(x)
            cls_x = self.clf(fx)

            return cls_x

        elif self.mode == 'train':
            if self.devices is not None:

                fx = self.extract_feature(x)
                cls_xf = self.clf(fx)
                # fx_added_cls = torch.cat([fx, cls_xf], dim=1)
                fxg = self.atten(fx, cls_xf)
                print(fxg.shape)
                fxg = fxg.reshape(fx.shape)
                print(fxg.shape)
                cls_xg = self.clf(fxg)

            # if self.devices is not None:
            #     inputs = (fx, fp.unsqueeze(0).repeat(len(self.devices), 1, 1))
            #     fxg, wx = nn.parallel.data_parallel(self.atten, inputs, device_ids=self.devices)
            # else:
            #     fxg, wx = self.atten(fx, fp.unsqueeze(0))
            #
            # cls_xf = self.clf(fx)
            # cls_xg = self.clf(fxg)

            return cls_xg, cls_xf, fx, fxg

        else:
            raise ValueError
