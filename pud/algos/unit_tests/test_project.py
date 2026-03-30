import torch
import unittest


class TestClass(unittest.TestCase):
    def setUp(self):
        self.Vmin = 0.0
        self.Vmax = 2.0
        self.probs = torch.FloatTensor([1.0, 0, 0, 0])

        self.N = self.probs.shape[0]
        self.dz = (self.Vmax - self.Vmin) / (self.N - 1)
        self.zs = torch.zeros(self.N)
        for i in range(self.N):
            self.zs[i] = self.Vmin + i * self.dz

    def compute_groundtruth(self, new_zs):
        target = torch.zeros_like(self.probs)
        new_zs = torch.clamp(new_zs, min=self.Vmin, max=self.Vmax)

        for j in range(target.shape[0]):
            bj = (new_zs[j] - self.Vmin) / self.dz
            lj = torch.floor(bj).long()
            uj = torch.ceil(bj).long()
            target[lj] += self.probs[j] * (uj + (uj == bj).to(torch.float) - bj).to(
                self.probs[j].dtype
            )
            target[uj] += self.probs[j] * (bj - lj).to(self.probs[j].dtype)
        return target

    def compute_in_batch(self, new_zs):
        new_zs = torch.clamp(new_zs, min=self.Vmin, max=self.Vmax)
        new_probs = torch.zeros_like(self.probs)

        bs = (new_zs - self.Vmin) / self.dz
        ls = torch.floor(bs).long()
        us = torch.ceil(bs).long()

        # When lower and upper adjacent class are the same, bump up the upper bound artificially by 1 to avoid
        # dumping the probabilities
        new_probs.scatter_add_(
            dim=-1,
            index=ls,
            src=self.probs
            * (us + (us == ls).to(torch.float) - bs).to(self.probs.dtype),
        )
        new_probs.scatter_add_(
            dim=-1, index=us, src=self.probs * (bs - ls).to(self.probs.dtype)
        )
        # Test case
        return new_probs

    def project_batch(self, new_zs):
        new_zs = torch.clamp(new_zs, min=self.Vmin, max=self.Vmax)
        new_probs = torch.zeros_like(new_zs)

        bs = (new_zs - self.Vmin) / self.dz
        ls = torch.floor(bs).long()
        us = torch.ceil(bs).long()

        # When lower and upper adjacent class are the same, bump up the upper bound artificially by 1 to avoid
        # dumping the probabilities
        out_of_range_fix = (us == ls).to(torch.float)
        new_probs.scatter_add_(
            dim=-1,
            index=ls,
            src=self.probs * (us + out_of_range_fix - bs).to(self.probs.dtype),
        )
        new_probs.scatter_add_(
            dim=-1, index=us, src=self.probs * (bs - ls).to(self.probs.dtype)
        )
        return new_probs

    def test_case_1(self):
        new_zs = self.zs + 0.5
        y = self.compute_groundtruth(new_zs)
        yp = self.project_batch(new_zs)
        self.assertTrue(torch.allclose(y, yp))

    def test_case_2(self):
        new_zs = self.zs + 1
        y = self.compute_groundtruth(new_zs)
        yp = self.project_batch(new_zs)
        self.assertTrue(torch.allclose(y, yp))

    def test_case_3(self):
        new_zs = self.zs + 2
        y = self.compute_groundtruth(new_zs)
        yp = self.project_batch(new_zs)
        self.assertTrue(torch.allclose(y, yp))

    def test_pytorch_easy_mistake(self):
        """
        Add a tensor to another tensor by indices that contains repeated index
        For example:
        inds = [0,0,0,0]
        x = [0,0,0,0]

        x[inds] = x[inds] + [1,0,0,0]

        The correct output is [1,0,0,0], as all elements are added to the first element,
        but it is easy to implement incorrectly as follows
        """
        inds = torch.LongTensor([0, 0, 0, 0])
        probs = torch.FloatTensor([1.0, 0.0, 0.0, 0.0])

        new_probs_wrong = torch.zeros_like(probs)
        new_probs_wrong[inds] += self.probs

        # Repeated inds will overwrite the targets, so it is not a sum, but the last element
        new_probs_wrong2 = torch.zeros_like(probs)
        new_probs_wrong2.scatter_(dim=0, index=inds, src=new_probs_wrong2[inds] + probs)

        # Missing operator to aggregate by inds
        # Add all elements according to the inds (containing repeated inds)
        # Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
        new_probs_correct = torch.zeros_like(probs)
        new_probs_correct.scatter_add_(
            dim=0, index=inds, src=probs + new_probs_correct[inds]
        )

        self.assertTrue(
            torch.allclose(new_probs_correct, torch.Tensor([1.0, 0.0, 0.0, 0.0]))
        )


if __name__ == "__main__":
    unittest.main()
