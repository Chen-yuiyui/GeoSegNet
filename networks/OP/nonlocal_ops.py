import torch
import torch.nn as nn
from torch.nn import functional as F



class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, neighbor_channel, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.neighbor_channel = neighbor_channel
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias,0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = conv_nd(in_channels=self.neighbor_channel, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.neighbor_channel, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            else:
                self.operation_function = self._dot_product

        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        else:
            raise NotImplementedError('Mode concatenation has not been implemented.')

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x, neighborFeat):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x, neighborFeat)
        return output

    def _embedded_gaussian(self, x, neighborFeat):
        batch_size = x.size(0)

        # print(x.size())
        # print(neighborFeat.size())
        # neighbor_channel = neighborFeat.size(1)
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # print('g_x', g_x.size())
        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        # theta_x = self.theta(neighborFeat)
        
        theta_x = self.theta(neighborFeat).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # print('theta_x',theta_x.size())
        phi_x = self.phi(neighborFeat).view(batch_size, self.inter_channels, -1)
        # print('phi_x',phi_x.size())
        f = torch.matmul(theta_x, phi_x)
        # print('f',f.size())
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, neighbor_channel, inter_channels=None, mode='embedded_gaussian', sub_sample=False, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              neighbor_channel,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

def nonlocalnet(input_layer,input_channel):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = NONLocalBlock3D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    else:
        net = NONLocalBlock3D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    return out



if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = NONLocalBlock1D(in_channels=128,neighbor_channel=3,mode='embedded_gaussian',sub_sample=False)
        x=torch.randn([32, 128, 1024])
        neighbor=torch.randn([32, 3, 1024])
        out = net(x, neighbor) 
        print(out.size())   
