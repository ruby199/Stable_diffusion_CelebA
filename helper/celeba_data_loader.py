import os
import zipfile
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader

def unzip_celeba_dataset(zip_file_path, extracted_images_path):
    if not os.path.exists(extracted_images_path):
        os.makedirs(extracted_images_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_images_path)

def define_condition():
    cond = 'Smiling'
    neg_cond = 'Male'
    return cond, neg_cond

class CelebADataset(Dataset):
    def __init__(self, img_dir, attributes_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_list = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, img))]
        self.attributes_df = pd.read_csv(attributes_file, sep='\s+', header=1, index_col=0)

        self.cond, self.neg_cond = define_condition()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label_1 = self.attributes_df.loc[img_name, self.cond]
        label_2 = self.attributes_df.loc[img_name, self.neg_cond]

        return image, (label_1, label_2)
    
def load_celeba_data(img_dir, attributes_file, batch_size=32, shuffle=True):
    transform = Compose([Resize((128, 128)), ToTensor()])
    dataset = CelebADataset(img_dir, attributes_file, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")  # Debugging line
    (cond, neg_cond) = define_condition()

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), (cond, neg_cond)

# Unzip the dataset (if needed)
zip_file_path = '../data/img_align_celeba.zip'
extracted_images_path = '../data/img_align_celeba'
unzip_celeba_dataset(zip_file_path, extracted_images_path)

# Path to attributes file
attributes_file_path = '../data/list_attr_celeba.txt'
