from utils.stnkd import STNkd
from utils.stn3d import STN3d
import torch
from torch import nn
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, (1,))
        self.conv2 = torch.nn.Conv1d(64, 128, (1,))
        self.conv3 = torch.nn.Conv1d(128, 1024, (1,))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
          # raise NotImplementedError("Feature Transformer not implemented.")
          self.fstn = STNkd(k=64)

    def forward(self, x, multi_resolution=None, use_max=None):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        trans_points = x
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # np.save("after_stn2.npy", x.detach().cpu().numpy())
        # exit(-1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # [batch_size, emb_size, num_points]
        x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, emb_size, num_points] ==> [batch_size, emb_size]
        x = x.view(-1, 1024)

        if self.global_feat:
            # 'x' are the global features: embedding vector which can be used for Classification or other tasks on the whole shape
            # Obtained by performing maxpooling on per-point features (see row 35)
            # Shape is: [batch_size, emb_size]
            return x, trans_points  # , trans, trans_feat
        else:
            # returning here the features of each point!
            # without maxpooling reduction
            # Shape is: [batch_size, num_points, emb_size]
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans_points # , trans, trans_feat