import torch
import torch.nn as nn

from feature_extractor import PointNetfeat
import torch.nn.functional as F


class Decoder(nn.Module):
    ''' Just a lightweight Fully Connected decoder:
    '''

    def __init__(self, num_points=2048):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


# our coments
# label = MUG
# 10, 1, 13
# 12, 21, 33
# ...

# output = encoder.forward(coords)
# output = 93891943843939438232

# point_cloud_output = decoder.forward(output)
# chamfer_loss between input and output

class PointNet_AutoEncoder(nn.Module):
    '''
    Complete AutoEncoder Model:
    Given an input point cloud X:
        - Step 1: encode the point cloud X into a latent low-dimensional code
        - Step 2: Starting from the code generate a representation Y as close as possible to the original input X

    Details:
    1. the 'code' size is hardocoded to 100 at line 45 - could be detrimental such a small code size
    1.1. If you change the code size you must modify accordingly also the decoder
    2. 'num_points' is the parameter controlling the number of points to be generated. In general we want to generate a number of points equal to the number of input points.
    '''

    def __init__(self, num_points=1024, feature_transform=False):
        super(PointNet_AutoEncoder, self).__init__()
        print("PointNet AE Init - num_points (# generated): %d" % num_points)

        # Encoder Definition
        self.encoder = torch.nn.Sequential(
            PointNetfeat(global_feat=True, feature_transform=feature_transform),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 100))

        # Decoder Definition
        self.decoder = Decoder(num_points=num_points)

    def forward(self, x):
        BS, N, dim = x.size()
        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding
        code = self.encoder(x)  # [BS, 3, N] => [BS, 100]

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_points]

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]

        return decoded