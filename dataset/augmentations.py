import random
from torchvision import transforms

def random_resize(base_size=112, variation=0.1):
    min_size = int(base_size * (1 - variation))
    max_size = int(base_size * (1 + variation))
    new_size = random.randint(min_size, max_size)
    return transforms.Resize((new_size, new_size))


def non_aug(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def base_aug(input_size):
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAdjustSharpness(2),
            transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
        ], p=0.7),
		
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def custom_aug(input_size):
    return transforms.Compose([
        transforms.RandomApply([
			random_resize(input_size//2, 0.1),
        ], p=0.7),
		
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def combined_aug(input_size):
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAdjustSharpness(2),
            transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
			random_resize(input_size//2, 0.1),
        ], p=0.7),
		
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

