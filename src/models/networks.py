import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], use_dropout = False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_dropout = use_dropout)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer, use_dropout = use_dropout)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
        
class IncontinuousLoss(nn.Module):
    def __init__(self, weight=1.):
        super(IncontinuousLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weight = weight
    def forward(self, x):
        loss = 0
        top = x[:,0,:,:]
        left = x[:,1,:,:]
        
        top_left = top[:,:,:-1]
        top_right = top[:,:,1:]
        loss += self.weight*self.criterion(top_left, top_right.detach())
        
        left_top = left[:,:-1,:]
        left_down = left[:,1:,:]
        loss += self.weight*self.criterion(left_top, left_down.detach())
        return loss
        
        
class LocalEnhancerLoss(nn.Module):
    def __init__(self, weight=1.):
        super(LocalEnhancerLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weight = weight
    def forward(self, x, y):
        loss = self.weight*self.criterion(x, y)
        return loss



##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='zero', use_dropout = False):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer, use_dropout = use_dropout).model
        #model_global = [model_global[i] for i in
        #                range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)
        self.output_nc = output_nc
        self.dropout = use_dropout
        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            model_refine = [EnhancerBlock(output_nc + input_nc, output_nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=self.dropout)]
            setattr(self, 'model' + str(self.n_local_enhancers) + "_" + str(n) + "_refine", nn.Sequential(*model_refine))

    def forward(self, input):
        ### build up one layer at a time
        import random
        if self.dropout:
            output_local_enhancers = random.randint(1, self.n_local_enhancers + 1)
        else:
            output_local_enhancers = self.n_local_enhancers + 1
        output_prev = self.model(input)
        for n_local_enhancers in range(1, output_local_enhancers):
            for i in range(4):
                model_refine = getattr(self, 'model' + str(self.n_local_enhancers) + "_" + str(n_local_enhancers) + "_refine")
                output_prev = model_refine(torch.cat((output_prev, input), dim=1))[:, :self.output_nc, :, : ]
        return output_prev

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, n = 1):
        super(UnetSkipConnectionBlock, self).__init__()
        self.meanDim = 16
        self.maxDim = 2
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc + 2 * (self.meanDim + self.maxDim), inner_nc, kernel_size=(8,2 + (n + 1)%2),
                             stride=(2, n%2 + 1), padding=(3,(n + 1)%2), bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(8, 2 + (n + 1)%2), stride=(2, n%2 + 1),
                                        padding=(3, (n + 1)%2))
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            skipConnectionModel = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=(8, 2 + (n + 1)%2), stride=(2, n%2 + 1),
                                        padding=(3, (n + 1)%2), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                skipConnectionModel = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                skipConnectionModel = down + [submodule] + up + [nn.Dropout(0.)]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=(8, 2 + (n + 1)%2), stride=(2, n%2 + 1),
                                        padding=(3, (n + 1)%2), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                skipConnectionModel = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                skipConnectionModel = down + [submodule] + up + [nn.Dropout(0.)]

        self.skipConnectionModel = nn.Sequential(*skipConnectionModel)

    def forward(self, x):
        mean_feat_hor = torch.mean(x[:, :self.meanDim,:, :], dim = 2, keepdim=True).expand_as(x[:, :self.meanDim,:, :])
        mean_feat_ver = torch.mean(x[:, :self.meanDim,:, :], dim = 3, keepdim=True).expand_as(x[:, :self.meanDim, :,:])
        max_feat_hor = torch.max(x[:, self.meanDim:self.meanDim + self.maxDim, :,:], dim = 2, keepdim=True)[0].expand_as(x[:, :self.maxDim,:, :])
        max_feat_ver = torch.max(x[:, self.meanDim:self.meanDim + self.maxDim, :,:], dim = 3, keepdim=True)[0].expand_as(x[:, :self.maxDim,:, :])
        if self.outermost:
            return self.skipConnectionModel(torch.cat((x, mean_feat_hor, mean_feat_ver, max_feat_hor, max_feat_ver), dim=1))
        else:
            return torch.cat([x, self.skipConnectionModel(torch.cat((x, mean_feat_hor, mean_feat_ver, max_feat_hor, max_feat_ver), dim=1))], 1)


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='zero',use_dropout = False):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.LeakyReLU(0.2, True)

        pre_model = [nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0), norm_layer(ngf), activation,
                 nn.Conv2d(ngf, ngf, kernel_size=(7,3), padding=(3,1)), norm_layer(ngf), activation,
                 nn.Conv2d(ngf, ngf, kernel_size=(7,3), padding=(3,1)), norm_layer(ngf), activation,
                 nn.Conv2d(ngf, ngf, kernel_size=(7,3), padding=(3,1)), norm_layer(ngf), activation]

        resnet = []
        ### resnet blocks
        for i in range(n_blocks):
            resnet += [ResnetBlockMean(min(ngf * 2 ** (n_downsampling), 1024), padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)]
        self.resnet = nn.Sequential(*resnet)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(min(ngf * 2 ** (n_downsampling - 1), 1024), min(ngf * 2 ** (n_downsampling), 1024), input_nc=None, submodule=self.resnet, norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        for i in range(n_downsampling, 2 , -1):
            unet_block = UnetSkipConnectionBlock(min(ngf * (2**(i - 2)), 1024), min(ngf * (2**(i - 1)), 1024), input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, n=i)
        unet_block = UnetSkipConnectionBlock(output_nc, 2 * ngf, input_nc=ngf, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_dropout=use_dropout)
        model = pre_model + [unet_block]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Define a resnet block

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d((2,1))]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d((2,1))]
        elif padding_type == 'zero':
            p = (2,1)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=(5,3), padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d((2,1))]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d((2,1))]
        elif padding_type == 'zero':
            p = (2,1)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=(5,3), padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
        
        
class EnhancerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, iter_conv_dim = 64):
        super(EnhancerBlock, self).__init__()
        self.mean_dim = output_dim
        self.output_dim = output_dim
        self.conv_block = self.build_conv_block(input_dim, output_dim, padding_type, norm_layer, activation, use_dropout, iter_conv_dim)
        self.use_dropout = use_dropout
        
    def build_conv_block(self, input_dim, output_dim, padding_type, norm_layer, activation, use_dropout, iter_conv_dim):
        conv_block = [norm_layer(input_dim + self.mean_dim * 2)]
        
        if use_dropout: 
            conv_block += [nn.Dropout(0.3)]
        else:
            conv_block += [nn.Dropout(0.)]
        conv_block += [nn.Conv2d(input_dim + self.mean_dim * 2, iter_conv_dim, kernel_size=(5,3), padding=(2, 1)),
                       norm_layer(iter_conv_dim),
                       activation]
        conv_block += [nn.Conv2d(iter_conv_dim, iter_conv_dim, kernel_size=(5,3), padding=(2,1)),
                       activation]             
        if use_dropout: 
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.)]
        conv_block += [nn.Conv2d(iter_conv_dim, iter_conv_dim, kernel_size=(1,1), padding=0)]
        conv_block += [nn.Conv2d(iter_conv_dim, output_dim, kernel_size=(1,1), padding=0)]
        
        conv_block+= [nn.Tanh()]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        
        x_augment = Variable(x, requires_grad=False)
        if self.use_dropout:
            x_augment = LocalAugment(x_augment)
        
        mean_feat_hor = torch.mean(x_augment[:, :self.mean_dim,:, :], dim = 2, keepdim=True).expand_as(x[:, :self.mean_dim,:, :])
        mean_feat_ver = torch.mean(x_augment[:, :self.mean_dim,:, :], dim = 3, keepdim=True).expand_as(x[:, :self.mean_dim, :,:])
        
        return torch.cat((self.conv_block(torch.cat((x_augment, mean_feat_hor, mean_feat_ver), dim=1)), x_augment[:, self.output_dim:,:, :]),dim=1)

def LocalAugment(input):
    import random
    # randomly pick a region from the whole table
    if random.random() > 0.3:
        # vertical region
        hor = False
        random_box_h = random.randint(1,30)
        random_box_w = random.randint(1, 2)
    else:
        # horizontal region
        hor = True
        random_box_h = random.randint(1,1)
        random_box_w = random.randint(1, 20)
        
    
    random_box_x = random.randint(1, input.size()[-2] - random_box_h - 1)
    random_box_y = random.randint(1, input.size()[-1] - random_box_w - 1)
    random_region_size = (input.size()[0], 1, random_box_h, random_box_w)
        
    if hor: 
        if random.random()>0.3:
            dim = 0
        else:
            dim = 4
    else: 
        if random.random()>0.3:
            dim = 1
        else:
            dim = 3
    
    # scale input from -1~1 to 0~1
    input[:,dim,:,:] = input[:,dim,:,:]/2.+0.5
    # randomly set 1 to 0 in the whole table
    input[:,dim,:,:] *= torch.round(torch.rand_like(input[:,dim,:,:], requires_grad=False) + 0.4)
    # scale input from 0~1 to -1~0
    input[:,dim,:,:] = input[:,dim,:,:] - 1
    # randomly set -1 to 0 in the whole table
    input[:,dim,:,:] *= torch.round(torch.rand_like(input[:,dim,:,:], requires_grad=False) + 0.4)
    # scale input back to 0~1 
    input[:,dim,:,:] = input[:,dim,:,:] + 1
    
    # randomly set 1s in the selected region
    if random.random()<0.2:input[:, dim, random_box_x:random_box_x + random_box_h, random_box_y:random_box_y + random_box_w] = torch.ones(random_region_size, requires_grad=False)
    # randomly set 0s in the selected region
    if random.random()<0.2:input[:, dim, random_box_x:random_box_x + random_box_h, random_box_y:random_box_y + random_box_w] = torch.zeros(random_region_size, requires_grad=False)
    # randomly set rands in the selected region
    input[:, dim, random_box_x:random_box_x + random_box_h, random_box_y:random_box_y + random_box_w] = torch.rand(random_region_size, requires_grad=False)
    # randomly round the values
    if random.random()>0.4: input[:, dim, random_box_x:random_box_x + random_box_h, random_box_y:random_box_y + random_box_w] = \
        torch.round(input[:, dim, random_box_x:random_box_x + random_box_h, random_box_y:random_box_y + random_box_w])
        
    input[:,dim,:,:] = 2 * input[:,dim,:,:] - 1.
    return input

class ResnetBlockMean(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.LeakyReLU(0.2, True), use_dropout=False):
        super(ResnetBlockMean, self).__init__()
        self.mean_dim = 32
        self.max_dim = 4
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)
        
    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        
        p = (2,1)
        
        conv_block += [nn.Conv2d(dim + (self.mean_dim + self.max_dim) * 2, dim, kernel_size=(5,3), padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=(5,3), padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        mean_feat_hor = torch.mean(x[:, :self.mean_dim,:, :], dim = 2, keepdim=True).expand_as(x[:, :self.mean_dim,:, :])
        mean_feat_ver = torch.mean(x[:, :self.mean_dim,:, :], dim = 3, keepdim=True).expand_as(x[:, :self.mean_dim, :,:])
        max_feat_hor = torch.max(x[:, self.mean_dim:self.mean_dim + self.max_dim, :,:], dim = 2, keepdim=True)[0].expand_as(x[:, :self.max_dim,:, :])
        max_feat_ver = torch.max(x[:, self.mean_dim:self.mean_dim + self.max_dim, :,:], dim = 3, keepdim=True)[0].expand_as(x[:, :self.max_dim,:, :])
        out = x + self.conv_block(torch.cat((x, mean_feat_hor, mean_feat_ver, max_feat_hor, max_feat_ver), dim=1))
        return out

    

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(2, stride=2, padding=(1, 1), count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = (5,3)
        padw = (2,1)
        
        kws = [(3,1),(7,3),(5,3)]
        strides = [1, 2, 2]
        padws = [(int(np.ceil((k[0] - 1.0) / 2)), int(np.ceil((k[1] - 1.0) / 2))) for k in kws]
        
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True),
                     nn.Conv2d(ndf, ndf, kernel_size=(5,3), stride=1, padding=(2,1)), nn.LeakyReLU(0.2, True),
                     nn.Conv2d(ndf, ndf, kernel_size=(5,3), stride=1, padding=(2,1)), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kws[n%len(kws)], stride= strides[n%len(kws)], padding=padws[n%len(kws)]),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
