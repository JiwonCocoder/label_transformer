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
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        fx, fp = f[:, :Nx], f[:, Nx:]

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w
class AttenHeadX_concat(nn.Module):
    def __init__(self, fdim, d_model, num_heads=8, num_classes=10, scaled=None):
        super().__init__()
        self.scaled = scaled
        self.nhead = num_heads
        self.d_model = d_model
        self.f_dim = fdim
        # self.intermediateDim = fdim + num_classes #(128 + 10)
        self.num_layers = 6
        #self.intermediateDim, self.num_layers
        # is based on Attention is All you Need
        self.embFC = nn.Linear(fdim + num_classes, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = self.nhead)
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
        fx_added_cls = torch.cat([fx, cls_xf], dim=1) #((bl+bu)*k, 138)
        proj_fx = self.embFC(fx_added_cls)

        if self.scaled == "yes":
            proj_fx_scaled = proj_fx * math.sqrt(self.d_model)
            # (in)((bl+bu)*k, 128+num_classes) (out) ((bl+bu)*k, 128)
            proj_fx_scaled = proj_fx_scaled.unsqueeze(0) #(1, (bl+bu)*k, 138)
            gx = self.transformer_encoder(proj_fx_scaled)
        # (in)((bl+bu)*k, 128+num_classes) (out) ((bl+bu)*k, 128)
        elif self.scaled == "no":
            proj_fx = proj_fx.unsqueeze(0) #(1, (bl+bu)*k, 138)
            gx = self.transformer_encoder(proj_fx)

        return gx


class AttenHeadX_pos_enc(nn.Module):
    def __init__(self, fdim, d_model, num_heads=8, num_classes=10, scaled=None):
        super().__init__()
        self.scaled = scaled
        self.nhead = num_heads
        # self.intermediateDim = fdim + num_classes #(128 + 10)
        self.num_layers = 6
        self.fdim = fdim
        self.d_model = d_model
        # self.intermediateDim, self.num_layers
        # is based on Attention is All you Need
        if self.fdim != self.d_model:
            print("fdim {} is not same as d_model {}". format(str(self.fdim), str(self.d_model)))
            self.embFC_feat = nn.Linear(self.fdim, self.d_model)
        self.embFC_class = nn.Linear(num_classes, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        # Encoder : in(1, S, 138), out(1, S, 138)
        # self.deEmbFC = nn.Linear(self.intermediateDim, fdim)
        '''
        TransformerEncoder is a stack of N encoder layers
        Args:
            encoder_layer: an instance of the TransformerEncoderLayer () class (required)
            num_layers: the number of sub-encoder-layer in the encoder(required)
            norm: the layer normalization component (optional)
        '''

    def forward(self, fx, cls_xf):
        if self.fdim != self.d_model:
            fx = self.embFC_feat(fx)
        #fx(embFC_feat_out) : (bs*(k+1), d_model)
        #cls_xf(embFC_class_in) : (bs*(k+1), 10)
        pos_enc_from_cls_xf = self.embFC_class(cls_xf)
        #pos_enc_from_cls_xf(embFC_class_out) : (bs*(k+1), d_model)
        fx_trans = fx + pos_enc_from_cls_xf
        if self.scaled == "yes":
            fx_trans_scaled = fx_trans * math.sqrt(self.d_model)
            fx_trans_scaled = fx_trans_scaled.unsqueeze(0)
            #fx_trans (trans_in) : (1, bs*(k+1), d_model)
            gx = self.transformer_encoder(fx_trans_scaled)
        elif self.scaled == "no":
            fx_trans = fx.unsqueeze(0)
            #fx_trans (trans_in) : (1, bs*(k+1), d_model)
            gx = self.transformer_encoder(fx_trans)
            #gx (trans_out) : (1, bs*(k+1), d_model)    


        return gx


class FeatMatch(nn.Module):
    def __init__(self, backbone, num_classes, devices, num_heads=1, amp=True,
                 attention='Feat', d_model = None, label_prop = None, detach = None, scaled = None):
        super().__init__()
        self.mode = 'train'
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.devices = devices
        self.default_device = torch.device('cuda', devices[0]) if devices is not None else torch.device('cpu')
        fext, self.fdim = make_backbone(backbone)
        self.detach = detach        
        print(self.devices)
        self.fext = nn.DataParallel(AmpModel(fext, amp), devices)
        print(attention)
        if d_model  == self.fdim:
            print("self.fdim {} is same as d_model{}".format(self.fdim, d_model))
            print("========================")
            print("correct")
            print("========================")
            self.d_model = self.fdim
        else:
            self.d_model = d_model
            self.deEmbFC = nn.Linear(d_model, self.fdim)
        if attention == 'no':
            print("init_baseline")

        elif attention == 'Feat':
            print("soft attention")
            self.atten = AttenHead(self.fdim, num_heads)
        elif attention == 'Transformer':
            if label_prop == 'concat':
                print("========================")            
                print("transformer_concat")
                print("========================")
                self.atten = AttenHeadX_concat(self.fdim, self.d_model, num_heads, num_classes, scaled)
            elif label_prop == 'pos_enc':
                print("========================")            
                print("transformer_pos_enc")
                print("========================")                
                self.atten = AttenHeadX_pos_enc(self.fdim, self.d_model, num_heads, num_classes, scaled)
        self.clf = nn.Linear(self.fdim, num_classes)

    def set_mode(self, mode):
        self.mode = mode

    def extract_feature(self, x):
        return self.fext(x)

    def forward(self, x):
        if self.mode == 'fext':
            return self.extract_feature(x)

        elif self.mode == 'pretrain':
            fx = self.extract_feature(x)
            cls_x = self.clf(fx)

            return cls_x

        elif self.mode == 'train':
            if self.devices is not None:

                fx = self.extract_feature(x) 
                #fx(clf_input) : (bs*(k+1), fdim)
                cls_xf = self.clf(fx)
                if self.detach == 'yes':
                    fx = fx.detach()
                    cls_xf = cls_xf.detach()
                #cls_xf(clf_out) : (bs*(k+1), num_class)
                fxg = self.atten(fx, cls_xf)
                #fxg(atten_out) : (1, bs*(k+1), fdim))
                fxg = fxg.squeeze(0)
                #fxg : (bs*(k+1), fdim)
                if self.fdim != self.d_model:
                    print("self.fdim {} is not same as self.d_model {}".format(self.fdim, self.fdim))
                    print("==================")
                    print("(error)deEmbedding is operating")
                    print("===================")
                    fxg = self.deEmbFC(fxg)
                cls_xg = self.clf(fxg)
                #cls_xg(cls_out) : (bs*(k+1), num_class)

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
