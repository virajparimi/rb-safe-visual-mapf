import torch
import unittest
from torch import nn


class TestBatchConv2d(unittest.TestCase):
    def test_batch_conv2d(self):
        net = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
        )
        im1 = torch.rand(4, 4, 64, 64)
        im2 = torch.rand(4, 4, 64, 64)
        cat_img = torch.cat([im1, im2], dim=0)

        cat_img.shape
        emb1 = net(im1)
        emb2 = net(im2)
        cat_emb = net(cat_img)

        torch.allclose(emb1, cat_emb[:4])
        torch.allclose(emb2, cat_emb[4:])


if __name__ == "__main__":
    unittest.main()
