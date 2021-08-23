from torch import nn
import torch
import torch.nn.functional as F


class PFNetDecoder(nn.Module):
    def __init__(self, resolutions):
        super(PFNetDecoder, self).__init__()
        self.resolutions = resolutions
        self.fc1 = nn.Linear(self.crop_point_num, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.resolutions[2] * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, self.resolutions[0], 3)  # 64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, self.resolutions[1], 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.resolutions[2] / 128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.resolutions[2], 3)

        return pc1_xyz, pc2_xyz, pc3_xyz
