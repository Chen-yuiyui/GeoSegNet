import numpy
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_utils

# ball_query = pointnet2_utils.BallQuery.apply
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Ball_query_and_group(torch.nn.Module):
    def __init__(self, radious, nsample):
        super(Ball_query_and_group, self).__init__()
        self.radious = radious
        self.nsample = nsample

    def forward(self, xyz, xyz_new):
        return pointnet2_utils.ball_query(self.radious,self.nsample,xyz,xyz)


if __name__ == '__main__':
    xyz = torch.rand(32, 1024, 3)*2-1
    xyz = xyz.to(device)
    print(torch.max(xyz))
    print(torch.min(xyz))
    print(xyz.size())
    starttime = datetime.datetime.now()
    # h = get_histogram(0.3, xyz)
    subPoints = pointnet2_utils.furthest_point_sample(xyz,512)
    print(subPoints.size())
    print(torch.max(subPoints))
    print(torch.min(subPoints))    
    # idx = torch.Tensor(idx)
    # grouped_xyz = pointnet2_utils.GroupingOperation(xyz, idx)
    # grouped_xyz = grouped_xyz.transpose(1, 2,0)
    endtime = datetime.datetime.now()
    print('TIME：',(endtime - starttime))
    # print(idx.size())
    # print(his.size())
    # print(grouped_xyz)
