from torch.utils.data import Dataset


class PointDataset(Dataset):
    def __init__(self, pc, normals):
        super().__init__()
        assert pc.shape[0] == normals.shape[0]
        self.pc = pc
        self.normals = normals

    def __getitem__(self, idx: int):
        return self.pc[idx, :3], normals[idx]

    def __len__(self):
        return self.pc.shape[0]
