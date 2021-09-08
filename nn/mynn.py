import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# A neural network with u_n， θ_c
# u_d = K(θ_c) u_n
# u_d(x) = \int K(x, y, θ_c) u_n(y) dy
class DirectNet_20(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_20, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DirectNet_50(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_50, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x




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
        # self.__initialize()
        
    def forward(self, x):
        x_branch, x_trunk = x[..., :self.branch_dim], x[..., -self.trunk_dim:]
        # x_branch = self.modus['Branch'](x_branch)
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)

        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias']
        
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
            
#     # def __initialize(self):
#     #     for i in range(1, self.trunk_depth):
#     #         self.weight_init_(self.modus['TrLinM{}'.format(i)].weight)
#     #         nn.init.constant_(self.modus['TrLinM{}'.format(i)].bias, 0)
# # preprocess the training data 

# class DirectData(Dataset):

#     def __init__(self, X, y):
        
#         self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
#         self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


# def build_bases(K, acc=0.9999, N_trunc=-1):

#     N_x, N_y, N_data = K.shape

#     data = K.reshape((-1, N_data))

#     # svd bases
#     u, s, vh = np.linalg.svd(np.transpose(data))
    
#     if N_trunc < 0:
#         s_sum_tot = np.dot(s, s)
#         s_sum = 0.0
#         for i in range(N_data):
#             s_sum += s[i]**2
#             if s_sum > acc*s_sum_tot:
#                 break
#         N_trunc = i+1
#     print("N_trunc = ", N_trunc)



#     scale = np.average(s[:N_trunc])
#     data_svd = u[:, 0:N_trunc] * s[:N_trunc]/scale
#     bases = vh[0:N_trunc, :]*scale

#     return data_svd, bases, N_trunc, s


