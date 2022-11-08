import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

TRAIN_DIR = "/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_2017_face_obfuscation"
WORKERS = 1

def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
            TRAIN_DIR,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=WORKERS, pin_memory=True, sampler=None)

    print("Loading iterator")
    loader_iter = iter(train_loader)
    print("Iterator loaded. Getting a data sample")
    sample = next(loader_iter)
    print("Finished")

if __name__ == '__main__':
    main()