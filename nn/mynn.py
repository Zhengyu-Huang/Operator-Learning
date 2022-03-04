import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import scipy.io
import h5py

import operator
from functools import reduce
from functools import partial


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)





class Module(torch.nn.Module):
    '''Standard module format. 
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d
    
    @dtype.setter    
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')
        
    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError
    
    @property        
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(StructureNN, self).__init__()
        
    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class FNN(StructureNN):
    '''Fully connected neural networks.
    '''
    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', softmax=False):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.softmax = softmax
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=-1)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules

class DeepONet(StructureNN):
    '''Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''

    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=50,
                 activation='relu'):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
        self.params = self.__init_params()

        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., -self.trunk_dim:]
        # x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']

    def trunk_forward(self, x):
        x_trunk = x[..., -self.trunk_dim:]
        
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return x_trunk
    
    def branch_forward(self, x):
        x_branch = x[..., :self.branch_dim]
        
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        
        return x_branch
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        # modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
        #                         self.activation, self.initializer)
        if self.branch_depth > 1:
            modules['BrLinM1'] = nn.Linear(self.branch_dim, self.width)
            modules['BrActM1'] = self.Act
            for i in range(2, self.branch_depth):
                modules['BrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['BrActM{}'.format(i)] = self.Act
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.width, self.width)
        else:
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.branch_dim, self.width)
            

        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params

class DeepFFONet(StructureNN):
    '''Deep full field operator network.
    Input: [batch size, branch_dim + trunk_dim * Np]
    Output: [batch size, 1]
    '''

    def __init__(self, branch_dim, trunk_dim, x_trunk, branch_depth=2, trunk_depth=3, width=50,
                 activation='relu'):
        super(DeepFFONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.x_trunk = x_trunk
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
        self.params = self.__init_params()

        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], self.x_trunk
        # x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        
        
        return torch.matmul(x_branch , x_trunk.T) + self.params['bias']

    def trunk_forward(self):
        x_trunk = self.x_trunk
        
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return x_trunk
    
    def branch_forward(self, x):
        x_branch = x[..., :self.branch_dim]
        
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        
        return x_branch
        
    def __init_modules(self):
        modules = nn.ModuleDict()
        # modules['Branch'] = FNN(self.branch_dim, self.width, self.branch_depth, self.width,
        #                         self.activation, self.initializer)
        if self.branch_depth > 1:
            modules['BrLinM1'] = nn.Linear(self.branch_dim, self.width)
            modules['BrActM1'] = self.Act
            for i in range(2, self.branch_depth):
                modules['BrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['BrActM{}'.format(i)] = self.Act
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.width, self.width)
        else:
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.branch_dim, self.width)
            

        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params
    
    
    
class PCAONet(StructureNN):
    '''PCA operator network.
    Input:  [batch size, branch_dim]
    Output: [batch size, field_size]
    '''

    def __init__(self, ind, outd, bases, layers=2, width=50, activation='relu', initializer='default'):
        super(PCAONet, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.bases = bases
        assert(bases.shape[0] == outd)
        
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        
    def forward(self, x):
        
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
        x = self.modus['LinMout'](x)
        x = torch.matmul(x , self.bases) + self.params['bias']
        
       
        return x
    
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd)
            
        return modules
        

            
    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params

################################################################
# fourier neural operator
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        

        self.fc1 = nn.Linear(self.width, 1)
        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        # if self.padding > 0:
        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        return x
    
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
# TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
# TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        

        self.fc1 = nn.Linear(self.width, 1)
      

    def forward(self, x):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    
    
# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c



