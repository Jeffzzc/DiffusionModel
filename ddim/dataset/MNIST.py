from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def create_mnist_dataset(data_path, batch_size, **kwargs):
    train = kwargs.get("train", True)  #指定是加载训练集 (train=True) 还是测试集 (train=False)。默认值是 True，表示加载训练集。
    download = kwargs.get("download", True) #指定是否从互联网上下载 MNIST 数据集。如果数据集已存在，则不会重新下载，默认值是 True。

    '''root=data_path：指定数据集的存储路径。
    train=train：指定是否加载训练集。
    download=download：是否从网络下载数据集。'''
    dataset = MNIST(root=data_path, train=train, download=download, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 的概率对图像进行随机水平翻转
        transforms.ToTensor(),                   # 将图像转换为 PyTorch tensor 格式，并将像素值归一化到 [0, 1] 的范围内。
        transforms.Normalize((0.5, ), (0.5, ))   # 使用均值 0.5 和标准差 0.5 对图像进行归一化。
    ]))

    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )
    
    '''shuffle：是否在每个 epoch 后打乱数据集。默认值是 True，通常用于训练集。
    drop_last：是否丢弃不满一个 batch 的最后一部分。默认值是 True，表示丢弃。
    pin_memory：是否将数据加载到 CUDA pinned memory 中。如果你使用 GPU 加速训练，设置为 True 会加速数据传输。
    num_workers：指定加载数据时使用的线程数。默认值是 4，表示使用 4 个线程加载数据。'''

    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)

    return dataloader