#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.OP import pointnet2_utils
import time
import scipy.io as sio
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def eigen_function(X):
    '''
    get eigen for a single point cloud neighbor feature
    :param X:  X is a Tensor, shape: [B, N, K, F]
    :return: KNN graph, shape: [B, N, F]
    '''
    B, N, K, F = X.shape
    # X_tranpose [N,F,K]
    device = X.device 
    X_tranpose = X.permute(0, 1, 3, 2)
    # high_dim_matrix [N, F, F]
    high_dim_matrix = torch.matmul(X_tranpose, X)

    high_dim_matrix = high_dim_matrix.cpu().detach().numpy()
    eigen, eigen_vec = np.linalg.eig(high_dim_matrix)
    # eigen_vec = torch.Tensor(eigen_vec).cuda()
    eigen = torch.Tensor(eigen).to(device)

    return eigen
def query_knn(k, x):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    return idx






def get_graph_feature_firstLayer(x, idx1, idx2, normalandRGB, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    org_xyz = x[:,0:3,:]

    org_feats = x[:,3:6,:]

#normal and rgb
    org_nr = normalandRGB.transpose(2, 1).contiguous()
    NR = org_nr.view(batch_size*num_points, -1)[idx1, :]
    NR = NR.view(batch_size, num_points, k, 3)


    org_xyz = org_xyz.transpose(2, 1).contiguous()
    xyz = org_xyz.view(batch_size*num_points, -1)[idx1, :]
    xyz = xyz.view(batch_size, num_points, k, 3)
    org_xyz = org_xyz.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1) 

    feat1 = torch.cat((xyz-org_xyz, org_xyz), dim=3)

    org_feats = org_feats.transpose(2, 1).contiguous()
    feats = org_feats.view(batch_size*num_points, -1)[idx2, :]
    feats = feats.view(batch_size, num_points, k, 3)
    org_feats = org_feats.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1) 

    feat2 = torch.cat((feats-org_feats, feats), dim=3)



    
    feature = torch.cat((feat1, feat2, NR), dim=3).permute(0, 3, 1, 2)
  
    return feature



def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, feature), dim=3).permute(0, 3, 1, 2)
  
    return feature

def get_graph_feature_self_neighbor(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature

def get_graph_feature_neighbor(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = (feature-x).permute(0, 3, 1, 2)
  
    return feature
def get_neighbor_distance(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature_distance = (feature**2+x**2-2*feature*x).permute(0, 3, 1, 2)
    feature_distance = torch.sum(feature_distance,dim=1)
    min_d = feature_distance.min(dim=2,keepdim=True)[0].repeat([1,1,k])
    max_d = feature_distance.max(dim=2,keepdim=True)[0].repeat([1,1,k])
    feature_distance = (feature_distance-min_d)/(max_d-min_d)
    return feature_distance

def get_neighbor_difference(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

 
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature_distance = (feature**2+x**2-2*feature*x).permute(0, 3, 1, 2)#B,C,N,K
    feature_distance = torch.sum(feature_distance,dim=1)#B,N,K
    feature_distance = torch.sum(feature_distance,dim=2)#B,N
    min_d = feature_distance.min(dim=1,keepdim=True)[0].repeat([1,num_points])
    max_d = feature_distance.max(dim=1,keepdim=True)[0].repeat([1,num_points])
    feature_distance = (feature_distance-min_d)/(max_d-min_d)
    return feature_distance

def h_points_back(new_points, H_points, H_idx):
    '''
    new_points [B,C,N]
    H_points [B,C,M]
    H_idx [B,M]
    '''
    B,C,N = new_points.shape
    _,_,M = H_points.shape
    device = torch.device('cuda')

    new_points = new_points.transpose(2, 1).contiguous().view(B*N, -1) #[B*N,C]
    H_points = H_points.transpose(2, 1).contiguous().view(B*M, -1)
    idx_base = torch.arange(0, B, device=device).view(-1, 1)*N

    H_idx = H_idx + idx_base

    H_idx = H_idx.view(-1)

    new_points[H_idx, :] = H_points

    new_points = new_points.view(B,N,C).permute(0,2,1).contiguous()
    return new_points


class IAF_PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(IAF_PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.H_mlp = nn.Sequential(nn.Conv1d(out_channel+13+8, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.out_mlp = nn.Sequential(nn.Conv1d(out_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.nonlocal_block = NONLocal_block(out_channel)

        self.conv5 = nn.Conv1d(out_channel, 32, 1)
        self.bn5 = nn.BatchNorm1d(32)
        self.drop5 = nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(32, 21, 1)


    def forward(self, xyz1, xyz2, points1, points2, rgb1 ,idx_1, k, lastpred):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
            lastpred: [B,S,13]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # print('xyz1',xyz1.size())
        # print('xyz2',xyz2.size())
        # print('points1',points1.size())
        # print('points2',points2.size())
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            interpolated_preds = torch.sum(index_points(lastpred, idx) * weight.view(B, N, 3, 1), dim=2)# [B, N, 13]

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        feat = new_points
        # diff_preds = get_neighbor_difference(interpolated_preds.permute(0,2,1).contiguous(), k=k, idx=idx_1) #B,N
        diff_points = get_neighbor_difference(new_points, k=k, idx=idx_1) #B,N
        # diff_xyz = get_neighbor_difference(xyz1.permute(0,2,1).contiguous(), k=k, idx=idx_1)
        # diff_rgb = get_neighbor_difference(rgb1, k=k, idx=idx_1)

        diff = diff_points 

        # print('DIFF',diff.size())

        M = int(N/4)
        outer = int(M*4/5)
        inner = int(M*1/5)
        outer_idx = diff.topk(k=outer, dim=-1)[1] #[B,M]
        inner_idx = torch.from_numpy(np.random.choice(N,inner,replace = False)).view(1,inner).repeat(B,1).cuda()

        H_idx = torch.cat((outer_idx,inner_idx), dim =1)
        H_points = torch.cat((new_points.permute(0, 2, 1).contiguous(),interpolated_preds),dim = -1)
        H_points = index_points(H_points, H_idx).permute(0, 2, 1).contiguous() #[B,C,M]
        H_points = self.H_mlp(H_points)
        # print('H_points',H_points.size())
        # new_points = h_points_back(new_points, H_points, H_idx)

        H_xyz = index_points(xyz1, H_idx) #[B,M,3]
        H_dists = square_distance(xyz1, H_xyz)
        # print(H_dists.size())
        # H_weight = torch.exp(-H_dists)
        # H_weight = F.softmax(H_weight, -1)
        new_points = self.nonlocal_block(new_points,H_points)



        new_points = self.out_mlp(new_points)

        x = self.drop5(F.relu(self.bn5(self.conv5(new_points))))
        x = self.conv6(x)
        x = F.log_softmax(x, dim=1)
        pred = x.permute(0, 2, 1)

        # H_xyz = index_points(xyz1, H_idx) #[B,N,3]
        # H_rgb = index_points(rgb1.permute(0,2,1), H_idx) #[B,N,3]

        H_xyz =None
        H_rgb =None
        return new_points, pred, H_xyz, H_rgb

class Self_Correlation(nn.Module):
    def __init__(self, in_channel):
        super(Self_Correlation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for i in range(3):
            self.mlp_convs.append(nn.Conv1d(in_channel, in_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(in_channel))

    def forward(self,feat):
        B, N, C = feat.shape
        new_points = feat
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        weight = torch.softmax(feat,1)
        new_points = feat + 0.5 * weight*feat
        return new_points

class Local_Correlation(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(Local_Correlation, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels 
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=1),
                bn(self.inter_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                             kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)



    def forward(self, x,idx,k):
        '''
        :param x: (b, c, n)
        :return:
        '''
        x = get_graph_feature_neighbor(x, k=k, idx=idx)

        b,c,n,k = x.shape
        g_x = self.g(x) 
        g_x = g_x.permute(0, 2, 3, 1).contiguous()#(b, n, k, c)

        theta_x = self.theta(x)
        theta_x = theta_x.permute(0, 2, 3, 1).contiguous() #(b, n, k, c)
        phi_x = self.phi(x).permute(0, 2, 1, 3).contiguous()#(b, n, c, k)

        f = torch.matmul(theta_x, phi_x)#(b, n, k, k)
        # N = f.size(-1)
        f_div_C = F.softmax(f, -1)

        y = torch.matmul(f_div_C, g_x) #(b, n, k, c)
        
        y = y.permute(0,3,1, 2).contiguous() #(b,c, n, k)
        out = self.W(y)
        out = out.max(dim=-1, keepdim=False)[0]#[B,C2,512]

        return out



class NONLocal_Correlation(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NONLocal_Correlation, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels 
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d


        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=1),
                bn(self.inter_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                             kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)



    def forward(self, x):
        '''
        :param x: (b, c, n)
        :return:
        '''
        b,c,n = x.shape
        g_x = self.g(x) 
        g_x = g_x.permute(0, 2 , 1).contiguous()#(b, n, c)

        theta_x = self.theta(x)
        theta_x = theta_x.permute(0, 2, 1).contiguous() #(b, n, c)
        phi_x = self.phi(x)#(b, c,n)

        f = torch.matmul(theta_x, phi_x)#(b, n,n)
        # N = f.size(-1)
        f_div_C = F.softmax(f, -1)

        y = torch.matmul(f_div_C, g_x) #(b, n,  c)
        
        y = y.permute(0,2,1).contiguous() #(b,c, n, k)
        out = self.W(y)
        
        return out



class NONLocal_block(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NONLocal_block, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels 
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        bn = nn.BatchNorm1d


        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                        kernel_size=1),
                bn(self.inter_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.inter_channels,
                             kernel_size=1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1)

    def forward(self, x, x_2):

        b,c,n = x.shape
        _,_,m = x_2.shape

        g_x = self.g(x_2).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)#[b,m,i]

        theta_x = self.theta(x).view(b, self.inter_channels, -1)#[B,i,n]
        theta_x = theta_x.permute(0, 2, 1)#[B,n,i]
        phi_x = self.phi(x_2).view(b, self.inter_channels, -1)#[B,i,m]
        f = torch.matmul(theta_x, phi_x)#[B,n,m]
        # f = f*H_weight
        f_div_C = F.softmax(f, -1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z




class Attentive_pool(nn.Module):
    def __init__(self, in_channel):
        super(Attentive_pool, self).__init__()
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # for i in range(2):
        #     self.mlp_convs.append(nn.Conv2d(in_channel, in_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(in_channel))
        self.mlp_convs = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.mlp_bns = nn.BatchNorm2d(in_channel)
        # for i in range(2):
        #     self.mlp_convs.append(nn.Conv2d(in_channel, in_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(in_channel))
    def forward(self, feat):
        '''
        feat1 : [B,C,N,K]
        '''
        B,C,N,K = feat.shape

        # weight = feat
        # for i, conv in enumerate(self.mlp_convs):
        # bn = self.mlp_bns[i]
        weight =  nn.LeakyReLU(negative_slope=0.2)( self.mlp_bns(self.mlp_convs(feat)))
        weight = F.softmax(weight,dim=-1)
        new_feat = torch.sum(feat*weight,dim=-1)

        return new_feat



def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)



class adaptive_feature_aggregation(nn.Module):
    def __init__(self, in_channel):
        super(adaptive_feature_aggregation, self).__init__()
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        for i in range(2):
            self.mlp_convs1.append(nn.Conv1d(in_channel, in_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm1d(in_channel))

        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        for i in range(2):
            self.mlp_convs2.append(nn.Conv1d(in_channel, in_channel, 1))
            self.mlp_bns2.append(nn.BatchNorm1d(in_channel))

    def forward(self, feat1,feat2):
        '''
        feat1 : [B,C,N]
        feat2 : [B,C,N]
        '''
        B,C,N = feat1.shape
        s_1 = feat1.mean(dim=-1,keepdim=True)
        s_2 = feat2.mean(dim=-1,keepdim=True)

        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            s_1 =  F.relu(bn(conv(s_1)))
        for i, conv in enumerate(self.mlp_convs2):
            bn = self.mlp_bns2[i]
            s_2 =  F.relu(bn(conv(s_2)))

        m_1 = torch.exp(s_1)/(torch.exp(s_1)+torch.exp(s_1))
        m_2 = torch.exp(s_2)/(torch.exp(s_2)+torch.exp(s_2))
        m_1 = m_1.view(B,C,1).repeat(1,1,N)
        m_2 = m_2.view(B,C,1).repeat(1,1,N)
        new_feat = feat1*m_1 + feat2*m_2

        return new_feat


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class ConvBNReLU2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU2D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes2D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes2D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm2d(channel),
                self.act,
                nn.Conv2d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x



class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)



class IAFNET(nn.Module):
    def __init__(self, args, output_channels=21):
        super(IAFNET, self).__init__()
        self.args = args
        self.k_layer1 = [32, 64]
        self.k_layer2 = [16, 32]
        self.k_layer3 = [16, 32]
        self.k_layer4 = [16, 24]
        self.k_layer5 = [8, 16]
        

        self.bng = nn.BatchNorm1d(1024)

        self.mlp_convs_layer1_scale1 = nn.ModuleList()
        self.mlp_convs_layer1_scale2 = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = 15
        for out_channel in [32,64]:
            self.mlp_convs_layer1_scale1.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            self.mlp_convs_layer1_scale2.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))

            last_channel = out_channel
        self.attpool_layer1_s1 = Attentive_pool(out_channel)
        self.attpool_layer1_s2 = Attentive_pool(out_channel)

        self.mlp_convs_layer2_scale1 = nn.ModuleList()
        self.mlp_convs_layer2_scale2 = nn.ModuleList()

        last_channel = 265+13+1+15
        for out_channel in [64,128]:
            self.mlp_convs_layer2_scale1.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            self.mlp_convs_layer2_scale2.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))

            last_channel = out_channel
        self.attpool_layer2_s1 = Attentive_pool(out_channel)
        self.attpool_layer2_s2 = Attentive_pool(out_channel)


        self.mlp_convs_layer3_scale1 = nn.ModuleList()
        self.mlp_convs_layer3_scale2 = nn.ModuleList()

        last_channel = 521+13+1+15
        for out_channel in [128,256]:
            self.mlp_convs_layer3_scale1.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            self.mlp_convs_layer3_scale2.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))

            last_channel = out_channel
        self.attpool_layer3_s1 = Attentive_pool(out_channel)
        self.attpool_layer3_s2 = Attentive_pool(out_channel)


        self.mlp_convs_layer4_scale1 = nn.ModuleList()
        self.mlp_convs_layer4_scale2 = nn.ModuleList()

        last_channel = 1033+13+1+15
        for out_channel in [256,512]:
            self.mlp_convs_layer4_scale1.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            self.mlp_convs_layer4_scale2.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))

            last_channel = out_channel
        self.attpool_layer4_s1 = Attentive_pool(out_channel)
        self.attpool_layer4_s2 = Attentive_pool(out_channel)

        self.mlp_convs_layer5_scale1 = nn.ModuleList()

        last_channel = 2057+13+1+15
        for out_channel in [512,1024]:
            self.mlp_convs_layer5_scale1.append(
                nn.Sequential(nn.Conv2d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            last_channel = out_channel

        self.attpool_layer5_s1 = Attentive_pool(out_channel)


        self.mlp_convs_layer5_groupAll = nn.ModuleList()
        last_channel = 1030+3
        for out_channel in [1024,512]:
            self.mlp_convs_layer5_groupAll.append(
                nn.Sequential(nn.Conv1d(last_channel, out_channel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channel),
                                   nn.LeakyReLU(negative_slope=0.2)))
            last_channel = out_channel





        self.conv_decoder5 = nn.Conv1d(out_channel*2, 32, 1)
        self.bn_decoder5  = nn.BatchNorm1d(32)
        self.drop_decoder5  = nn.Dropout(0.5)
        self.conv_decoder5_2  = nn.Conv1d(32, output_channels, 1)


        self.fp5 = IAF_PointNetFeaturePropagation(2060+6, [1024, 512])
        self.fp4 = IAF_PointNetFeaturePropagation(774+3, [512, 256])
        self.fp3 = IAF_PointNetFeaturePropagation(390+3, [256, 128])
        self.fp2 = IAF_PointNetFeaturePropagation(198+3, [128, 64])

        self.self_cor = Self_Correlation(64)
        self.local_cor = Local_Correlation(64)
        self.AFA_1 = adaptive_feature_aggregation(64)
        self.nonlocal_cor = NONLocal_Correlation(64)
        self.AFA_2 = adaptive_feature_aggregation(64)


        self.conv5 = nn.Conv1d(64, 32, 1)
        self.bn5 = nn.BatchNorm1d(32)
        self.drop5 = nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(32, output_channels, 1)

        self.pointmlp1 = ConvBNReLURes2D(265)
        self.pointmlp2 = ConvBNReLURes2D(521)
        self.pointmlp3 = ConvBNReLURes2D(1033)
        self.pointmlp4 = ConvBNReLURes2D(2057)
        self.nr_mlp =  ConvBNReLURes2D(6)

        for i in range(5):
            out_channel1 = [64,128,256,512,1024]
            # pre_block_num = pre_blocks[i]
            pos_block_num = 3
            res_expansion = 1.0
            pos_block_module = PosExtraction(out_channel1[i], pos_block_num, groups=1,
                                             res_expansion=res_expansion, bias=True, activation="relu")
            self.pos_blocks_list.append(pos_block_module)

           # last_channel = out_channel1[i]

    def eigen_Net(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        device = torch.device('cuda')
        saveidx = idx

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        neighbor_xyz = self.gather_neighbour(x, saveidx)  # batch*npoint*nsamples*3
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)  # gather neighbour
        relative_info, relative_alpha, relative_beta, geometric_dis, local_volume = self.relative_pos_transforming(x, saveidx ,neighbor_xyz)
        neighbor_mean = torch.mean(neighbor_xyz, dim=-2)

        direction = x - neighbor_mean
        direction_tile = direction.unsqueeze(2).repeat(1, 1, saveidx.shape[-1], 1)
        direction_alpha = torch.atan2(direction_tile[:,:,:,1],direction_tile[:,:,:,0]).unsqueeze(dim=-1)
        direction_xydis = torch.sqrt(torch.sum(torch.square(direction_tile[:,:,:,:2]), dim=-1))
        direction_beta = torch.atan2(direction_tile[:,:,:,2], direction_xydis).unsqueeze(dim=-1)

        # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        angle_alpha = relative_alpha - direction_alpha
        angle_beta = relative_beta - direction_beta
        angle_updated = torch.cat([angle_alpha, angle_beta], dim=-1)
        local_rep = torch.cat([angle_updated, relative_info], dim=-1)

        eigen = eigen_function(feature - x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)) #b n c
        # eigen_vec = eigen_vec.reshape([batch_size, num_points, -1])
        # 计算K近邻的特征矩阵的特征值
        feature = torch.cat((x, eigen), dim=2) #b n c

        global_dis = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdims=True))
        global_dis,_= torch.max(global_dis, dim=-1)
        global_volume = torch.pow(global_dis, 3)
        lg_volume_ratio = (local_volume / global_volume).unsqueeze(-1)

        # concat
        idx2 = knn(eigen.permute(0, 2, 1), k=k)  # (batch_size, num_points, k)

        idx_base2 = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx2 = idx2 + idx_base2

        idx2 = idx2.view(-1)

        return feature.permute(0, 2, 1), idx, idx2, local_rep, lg_volume_ratio, geometric_dis

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
            # gather the coordinates or features of neighboring points
            batch_size = pc.shape[0]
            num_points = pc.shape[1]
            d = pc.shape[2]
            index_input = neighbor_idx.reshape(batch_size, -1)
            features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
            features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1],
                                        d)  # batch*npoint*nsamples*channel
            return features

    def relative_pos_transforming(self, xyz, neigh_idx,neighbor_xyz):

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz

        relative_alpha = torch.atan2(relative_xyz[:, :, :, 1], relative_xyz[:, :, :, 0]).unsqueeze(-1)
        relative_xydis = torch.sqrt(torch.sum(torch.square(relative_xyz[:, :, :, :2]), dim=-1))
        relative_beta = torch.atan2(relative_xyz[:, :, :, 2], relative_xydis).unsqueeze(-1)
        relative_dis = torch.sqrt(torch.sum(torch.square(relative_xyz), dim=-1, keepdims=True))

        relative_info = torch.cat([relative_dis, xyz_tile, neighbor_xyz], axis=-1)

        # negative exp of geometric distance
        exp_dis = torch.exp(-relative_dis)

        # volume of local region
        relative_dis, _ = torch.max(relative_dis,dim=-1,keepdim=False)
        relative_dis, _ = torch.max(relative_dis, dim=-1, keepdim=False)
        local_volume = torch.pow(relative_dis, 3)

        return relative_info, relative_alpha, relative_beta, exp_dis, local_volume

    def conv_MEG(self, xyz, nr, feats, k, mlp_convs, att_pool, pointmlps , nr_mlp ,firstLayer=False,Layer=0,groupALL=False):
        

        if groupALL==True:
            # print('xyz',xyz.size())
            # print('nr',nr.size())
            # print('feats',feats.size())
            x = torch.cat((xyz.permute(0,2,1).contiguous(),nr,feats),dim=1)

        else:
            x, idx0, idx1, local_rep, lg_volume_ratio, geometric_dis= self.eigen_Net(xyz.permute(0,2,1).contiguous(), k=k) # 仅仅使用xyz坐标计算出特征值
            if firstLayer==True:
                x = get_graph_feature_firstLayer(x, idx0, idx1, nr, k=k) # 第一层的特征,其余层使用上一层特征

            else:
                # idx_FKNN = query_knn(k, feats)
                xyz_group = get_graph_feature_self_neighbor(xyz.permute(0,2,1).contiguous(), k=k, idx=idx0) #坐标聚合
                NR_group = get_graph_feature_neighbor(nr, k=k, idx=idx0)# normal and rgb  聚合
                # NR_var = torch.var(NR_group, dim=3 ,keepdim=True).repeat(1,1,1,k)
                # NR_group = torch.cat((NR_group, NR_var), dim = 1)
                # NR_group = nr_mlp(NR_group)
                # #local_rep 过一个mlp 或者两个mlp 然后concat local_rep feats
                x_knn1 = get_graph_feature(feats, k=k, idx=idx0)#xyz感受野1
                x_knn2 = get_graph_feature(feats, k=k, idx=idx1)#eigenvalue感受野2
                #x1 = torch.cat((x_knn1,x_knn2,local_rep.permute(0,3,1,2)), dim = 1)
                #x1 = pointmlps(x1)
                f_gc = torch.cat([xyz,lg_volume_ratio],dim=-1).unsqueeze(-2).repeat(1, 1, k, 1)#concat xyz, lg_volume_ratio
                #f_gc add a mlp then concat
                x = torch.cat((xyz_group, NR_group, x_knn1,x_knn2,local_rep.permute(0,3,1,2),
                               f_gc.permute(0,3,1,2), geometric_dis.permute(0,3,1,2)),dim = 1) #concat所有特征

        for i, conv in enumerate(mlp_convs):
            x =  mlp_convs[i](x)

        if groupALL==False:
            # x = x.max(dim=-1, keepdim=False)[0]#[B,C2,512]
            x = att_pool(x)
        else:
            return x
        # if firstLayer==True:
        #     return x, idx0
        return x, idx0






    def forward(self,  normalandRGB, x, seg, return_features=False ):
        # print('NR',normalandRGB.size())
        # print('x',x.size())        
        normalandRGB = normalandRGB.permute(0,2,1).contiguous()
        x = x.permute(0,2,1).contiguous()
        batch_size = x.size(0)
        num_points_1 = x.size(2)  #4096
        num_points_2 = int(num_points_1/4) #1024
        num_points_3 = int(num_points_1/16) #256
        num_points_4 = int(num_points_1/64) #64
        num_points_5 = int(num_points_1/256) #16

        indexs = []

        seg = seg.view(batch_size, 1, num_points_1).float().contiguous()

        ######BLOCK1---------------------------------------------
        N1_points = x.permute(0,2,1).contiguous()#[B,4096,3]
        coords = [N1_points]
        x1_s1,idx_1 = self.conv_MEG( N1_points, normalandRGB, None, self.k_layer1[0], self.mlp_convs_layer1_scale1,
            self.attpool_layer1_s1,None,None, firstLayer=True)
        x1_s2,_ = self.conv_MEG( N1_points, normalandRGB, None, self.k_layer1[1], self.mlp_convs_layer1_scale2,
            self.attpool_layer1_s2,None,None, firstLayer=True)

        x1 = x1_s1+x1_s2
       # save_x1 = x1
        x1 = self.pos_blocks_list[0](x1)
        ######BLOCK2---------------------------------------------
        fps_id_2 = pointnet2_utils.furthest_point_sample(N1_points, num_points_2)
        indexs.append(fps_id_2)
        NR_downsample_2 = (
            pointnet2_utils.gather_operation(
                normalandRGB, fps_id_2)
            )
        seg_downsample_2 = (
            pointnet2_utils.gather_operation(
                seg, fps_id_2)
            ) 
        N2_points = (
            pointnet2_utils.gather_operation(
                N1_points.transpose(1, 2).contiguous(), fps_id_2
            ).transpose(1, 2).contiguous())#[B,512,3]
        coords.append(N2_points)
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1, fps_id_2)
            )#[B,C1,512]

        x2_s1,idx_2 = self.conv_MEG( N2_points, NR_downsample_2, x1_downSample, self.k_layer2[0], self.mlp_convs_layer2_scale1,
            self.attpool_layer2_s1,self.pointmlp1,self.nr_mlp, Layer=2)
        x2_s2,_ = self.conv_MEG( N2_points, NR_downsample_2, x1_downSample, self.k_layer2[1], self.mlp_convs_layer2_scale2,
            self.attpool_layer2_s2,self.pointmlp1,self.nr_mlp)
        # x2_s3 = self.conv_MEG( N2_points, NR_downsample_2, x1_downSample, self.k_layer2[2], self.mlp_convs_layer2_scale3)
        # x2 = torch.cat((x2_s1,x2_s2),dim=1)
        x2 = x2_s1 + x2_s2
        x2 = self.pos_blocks_list[1](x2)
       # save_x2 = x2
        ######BLOCK3---------------------------------------------

        fps_id_3 = pointnet2_utils.furthest_point_sample(N2_points, num_points_3)
        indexs.append(fps_id_3)
        N3_points = (
            pointnet2_utils.gather_operation(
                N2_points.transpose(1, 2).contiguous(), fps_id_3
            ).transpose(1, 2).contiguous())#[B,256,3]
        coords.append(N3_points)
        x2_downSample = (
            pointnet2_utils.gather_operation(
                x2, fps_id_3)
            )#[B,C1,256]
        seg_downsample_3 = (
            pointnet2_utils.gather_operation(
                seg_downsample_2, fps_id_3)
            )  
        NR_downsample_3 = (
            pointnet2_utils.gather_operation(
                NR_downsample_2, fps_id_3)
            )

        x3_s1, idx_3 = self.conv_MEG( N3_points, NR_downsample_3, x2_downSample, self.k_layer3[0], self.mlp_convs_layer3_scale1,
            self.attpool_layer3_s1,self.pointmlp2,self.nr_mlp,Layer=3)
        x3_s2, _ = self.conv_MEG( N3_points, NR_downsample_3, x2_downSample, self.k_layer3[1], self.mlp_convs_layer3_scale2,
            self.attpool_layer3_s2,self.pointmlp2,self.nr_mlp)
        # x3_s3 = self.conv_MEG( N3_points, NR_downsample_3, x2_downSample, self.k_layer3[2], self.mlp_convs_layer3_scale3)
        # x3 = torch.cat((x3_s1,x3_s2),dim=1)
        x3 = x3_s1 + x3_s2
        x3 = self.pos_blocks_list[2](x3)
       # save_x3 = x3

        ######BLOCK4---------------------------------------------

        fps_id_4 = pointnet2_utils.furthest_point_sample(N3_points, num_points_4)
        indexs.append(fps_id_4)
        N4_points = (
            pointnet2_utils.gather_operation(
                N3_points.transpose(1, 2).contiguous(), fps_id_4
            ).transpose(1, 2).contiguous())#[B,256,3]
        coords.append(N4_points)
        x3_downSample = (
            pointnet2_utils.gather_operation(
                x3, fps_id_4)
            )#[B,C1,256]
        seg_downsample_4 = (
            pointnet2_utils.gather_operation(
                seg_downsample_3, fps_id_4)
            )  
        NR_downsample_4 = (
            pointnet2_utils.gather_operation(
                NR_downsample_3, fps_id_4)
            )

        x4_s1, idx_4 = self.conv_MEG( N4_points, NR_downsample_4, x3_downSample, self.k_layer4[0], self.mlp_convs_layer4_scale1,
            self.attpool_layer4_s1,self.pointmlp3,self.nr_mlp,Layer=4)
        x4_s2, _ = self.conv_MEG( N4_points, NR_downsample_4, x3_downSample, self.k_layer4[1], self.mlp_convs_layer4_scale2,
            self.attpool_layer4_s2,self.pointmlp3,self.nr_mlp)

        # x4 = torch.cat((x4_s1,x4_s2),dim=1)
        x4 = x4_s1 + x4_s2
        x4 = self.pos_blocks_list[3](x4)
       # save_x4 = x4



        ######BLOCK5---------------------------------------------

        fps_id_5 = pointnet2_utils.furthest_point_sample(N4_points, num_points_5)
        indexs.append(fps_id_5)
        N5_points = (
            pointnet2_utils.gather_operation(
                N4_points.transpose(1, 2).contiguous(), fps_id_5
            ).transpose(1, 2).contiguous())#[B,256,3]
        coords.append(N5_points)
        x4_downSample = (
            pointnet2_utils.gather_operation(
                x4, fps_id_5)
            )#[B,C1,256]
        seg_downsample_5 = (
            pointnet2_utils.gather_operation(
                seg_downsample_4, fps_id_5)
            )  
        NR_downsample_5 = (
            pointnet2_utils.gather_operation(
                NR_downsample_4, fps_id_5)
            )

        x5_s1,idx_5 = self.conv_MEG( N5_points, NR_downsample_5, x4_downSample, self.k_layer5[0], self.mlp_convs_layer5_scale1,
            self.attpool_layer5_s1,self.pointmlp4,self.nr_mlp,Layer=5)
        x5 = x5_s1
        x5 = self.pos_blocks_list[4](x5)
       # save_x5 = x5




        pred5 = self.drop_decoder5(F.relu(self.bn_decoder5(self.conv_decoder5(x5))))
        pred5 = self.conv_decoder5_2(pred5)
        pred5 = F.log_softmax(pred5, dim=1)
        pred5 = pred5.permute(0, 2, 1)

        x_g = self.conv_MEG( N5_points, NR_downsample_5, x5, self.k_layer5[0], self.mlp_convs_layer5_groupAll,None,None,None, groupALL=True)
        x_g = x_g.max(dim=-1, keepdim=False)[0]
        x_g = x_g.contiguous().view(batch_size,x_g.size(1),1).repeat(1,1,num_points_5) #b 512 16


        ######Decoder-----------------------------------------------------------------------
        x5 = torch.cat([x_g, N5_points.permute(0,2,1).contiguous(), NR_downsample_5, x5], dim=1)

        x4 = torch.cat([N4_points.permute(0,2,1).contiguous(), NR_downsample_4, x4], dim=1)
        x4,pred4,H_xyz4, H_rgb4 = self.fp5(N4_points, N5_points, x4, x5, NR_downsample_4, idx_4, self.k_layer4[0], pred5)

        x3 = torch.cat([N3_points.permute(0,2,1).contiguous(), NR_downsample_3, x3], dim=1)
        x3,pred3,H_xyz3, H_rgb3 = self.fp4(N3_points, N4_points, x3, x4, NR_downsample_3, idx_3, self.k_layer3[0], pred4)
        #########################
        x2 = torch.cat([N2_points.permute(0,2,1).contiguous(), NR_downsample_2, x2], dim=1)
        x2,pred2,H_xyz2, H_rgb2 = self.fp3(N2_points, N3_points, x2, x3, NR_downsample_2, idx_2, self.k_layer2[0], pred3)
        
        # g_x = g_x.view(batch_size, g_x.size()[1], 1).repeat(1, 1, x1.size()[2])
        x1 = torch.cat([ N1_points.permute(0,2,1).contiguous(), normalandRGB, x1], dim=1)
        x1,pred1,H_xyz1, H_rgb1 = self.fp2(N1_points, N2_points, x1, x2, normalandRGB, idx_1, self.k_layer1[0], pred2)

        feats = [x1.permute(0, 2, 1), x2.permute(0, 2, 1), x3.permute(0, 2, 1), x4.permute(0, 2, 1),
                 x5.permute(0, 2, 1)]
        ## correlation
        x_self = self.self_cor(x1)
        x_local = self.local_cor(x_self,idx_1,self.k_layer1[0])
        x = self.AFA_1(x_self,x_local)
        #x_after_local = x
        x_nonlocal = self.nonlocal_cor(x)
        x = self.AFA_2(x,x_nonlocal)
        #x_after_nonlocal = x
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        x = self.conv6(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)



        seg_downsample_2 = torch.squeeze(seg_downsample_2).long()
        seg_downsample_3 = torch.squeeze(seg_downsample_3).long()
        seg_downsample_4 = torch.squeeze(seg_downsample_4).long()
        seg_downsample_5 = torch.squeeze(seg_downsample_5).long()

        if return_features:
            return x, x1
        else:
            return x, pred1,pred2,pred3,pred4,pred5,seg_downsample_2,seg_downsample_3,seg_downsample_4, seg_downsample_5,coords,indexs,feats


class Loss(nn.Module):
    def __init__(self):
        """
        初始化
        """
        super(Loss, self).__init__()
        self.nsamples = [64, 32, 16, 8, 4]
        self.temperature = 1.0

    def stats_boundary_loss(self, coords, labels, feats, num_classes):
        """
        边界损失函数 L_cbl
        :param coords: 原始坐标和4次下采样的坐标，用于计算邻域
        :param labels: 原始标签
        :param feats: decoder中由points经MLP生成，32通道
        :param num_classes: 语义类别
        :return: loss
        """
        loss = torch.tensor(.0).to('cpu')
        labels = F.one_hot(labels.to(torch.int64), num_classes)  # [B, N, 13]
        for i, (coord, feat) in enumerate(zip(coords, feats)):

            # 生成场景标签
            if i == 0:
                probability_labels = labels
            else:
                kneighbors = i * 4  # K个邻居, 1, 4, 16, 64, 256
                neighbor_indexs = knn_point(kneighbors, coords[0], coord)  # [B, S, K], S: number of coords[i]
                probability_labels = index_points(labels, neighbor_indexs)  # [B, S, K, 13]
                probability_labels = probability_labels.float().mean(dim=-2)  # [B, S, 13]

            # 对标签和特征选择nsample邻域
            nsample = self.nsamples[i]
            neighbor_indexs = knn_point(nsample, coord, coord)  # [B, S, nsample], S: number of coords[i]
            neighbor_labels = index_points(probability_labels, neighbor_indexs)  # [B, S, K, 13]
            neighbor_feats = index_points(feat, neighbor_indexs)  # [B, S, K, C]

            # 查找边界点
            center_label = torch.argmax(torch.unsqueeze(probability_labels, -2), -1)  # [B, S, 1, 13] -> [B, S, 1]
            neighbor_labels = torch.argmax(neighbor_labels, -1)  # [B, S, K]
            mask = center_label == neighbor_labels  # [B, S, K], bool
            point_mask = torch.sum(mask.int(), -1)  # [B, S], 每个点的邻居有多少是和自己标签相同的
            point_mask = (point_mask > 0) & (point_mask < nsample)  # [B, S], bool

            # 没有边界点则直接进入下一个stage
            if not torch.any(point_mask):
                continue

            # 获得边界点特征
            mask = mask[point_mask]  # [B, sub_S, K]
            feat = feat[point_mask]  # [B, sub_S, C]
            neighbor_feats = neighbor_feats[point_mask]  # [B, sub_S, K, C]

            # 计算loss
            dist = torch.unsqueeze(feat, -2) - neighbor_feats  # [B, sub_S, 1, C] - [B, sub_S, K, C]
            dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + 1e-6)  # [B, sub_S, K, C] -> [B, sub_S, K]
            dist = - dist
            dist = dist - torch.max(dist, -1, keepdim=True)[0]
            dist = dist / self.temperature

            exp = torch.exp(dist)  # [B, sub_S, K]
            pos = torch.sum(exp * mask.float())
            neg = torch.sum(exp)

            loss = loss - torch.log(pos / (neg + 1e-6))

        return loss

    def stats_ordinary_loss(self, labels, indexs, preds, num_classes):
        """
        计算从decoder输出的5个预测结果preds的损失
        :param labels: 原始标签
        :param indexs: encoder每个stage下采样的点的下标, [B, N/4], ... , [B, N/256]
        :param preds: decoder每个stage生成的预测标签, [B, N, 13], ... , [B, N/64, 13], [B, N/256, 13]
        :param num_classes: 语义类别，默认13
        :return: loss
        """
        pred1, pred2, pred3, pred4, pred5 = preds
        label2, label3, label4, label5 = [labels.gather(dim=1, index=indexs[i]) for i in range(len(indexs))]

        loss1 = F.cross_entropy(pred1.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        loss2 = F.cross_entropy(pred2.contiguous().view(-1, num_classes), label2.contiguous().view(-1))
        loss3 = F.cross_entropy(pred3.contiguous().view(-1, num_classes), label3.contiguous().view(-1))
        loss4 = F.cross_entropy(pred4.contiguous().view(-1, num_classes), label4.contiguous().view(-1))
        loss5 = F.cross_entropy(pred5.contiguous().view(-1, num_classes), label5.contiguous().view(-1))

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        return loss

    def forward(self, labels, coords, feats, num_classes=21, alpha=0.2, beta=0.2):
        """
        :param labels: 输入的标签, B x N x 1
        :param indexs: 每次FPS下采样所采点的集合, [index2, index3, index4, index5]
        :param output: 模型预测结果, B x N x 13
        :param preds: 模型5个stage的预测标签, [pred1, pred2, pred3, pred4, pred5]
        :param coords: 5个stage中点的坐标, [xyz1, xyz2, xyz3, xyz4, xyz5]
        :param feats: 5个stage中特征, [feat1, feat2, feat3, feat4, feat5]
        :param num_classes: 语义标签种类
        :param alpha: 边界损失权重
        :param beta: 普通点损失权重
        :return: loss
        """
        #output_loss = F.cross_entropy(output.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
        boundary_loss = self.stats_boundary_loss(coords, labels, feats, num_classes)
        #ordinary_loss = self.stats_ordinary_loss(labels, indexs, preds, num_classes)

        #loss = output_loss + alpha * boundary_loss + beta * ordinary_loss
        return boundary_loss