import logging
from PIL import Image


from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .dataset import CUB, NDTWIN
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.dataset == 'CUB_200_2011':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    elif args.dataset == 'NDTWIN':
        # Improved preprocessing for face verification
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size + 32, args.img_size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = NDTWIN(root=args.data_root, pair_txt='train_pairs_dataset.txt', transform=train_transform)
        testset = NDTWIN(root=args.data_root, pair_txt='test_pairs_dataset.txt', transform=test_transform)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=2,  # Reduced for single GPU
                              drop_last=True,
                              pin_memory=True,
                              prefetch_factor=2)  # Added for better performance
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,  # Reduced for single GPU
                             pin_memory=True,
                             prefetch_factor=2) if testset is not None else None

    return train_loader, test_loader
