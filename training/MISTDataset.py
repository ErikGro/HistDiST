from typing import List
from pathlib import Path
from datasets import Dataset
from random import shuffle
from PIL import Image
from torchvision import transforms

from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

class MISTDataset(Dataset):
    def __init__(self, path: str, num_samples_to_use: int):
        self.path = path
        
        heImagePaths = list(Path(path + "/TrainValAB/trainA").glob('*.jpg'))

        shuffle(heImagePaths)
        if num_samples_to_use is not None:
            self.heImagePaths = heImagePaths[:num_samples_to_use]
        else:
            self.heImagePaths = heImagePaths

    def __len__(self) -> int:
        return len(self.heImagePaths)
    
    def _indices():
        return 

    def __getitem__(self, i: int) -> dict:
        heImagePath = str(self.heImagePaths[i])
        ihcImagePath = heImagePath.replace("trainA", "trainB")

        toTensor = transforms.ToTensor()

        heImage = toTensor(Image.open(heImagePath).convert("RGB"))
        ihcImage = toTensor(Image.open(ihcImagePath).convert("RGB"))

        return dict(
            he_image=heImage,
            ihc_image=ihcImage
        )
    
    def __getitems__(self, indices: List[int]):
        return list(map(lambda x: self[x], indices))
    

def gen_examples(path, num_samples_to_use):
    def fn():
        for he_path in list(Path(path + "/TrainValAB/trainA").glob('*.jpg'))[:num_samples_to_use]:
            yield {
                "he_image": {"path": str(he_path)},
                "ihc_image": {"path": str(he_path).replace("trainA", "trainB")}
            }

    return fn

mist_ds = Dataset.from_generator(
    # /graphics/scratch2/students/grosskop/raw_datasets/ER 4096
    # /graphics/scratch2/students/grosskop/BCI_dataset 3888
    # /graphics/scratch2/students/grosskop/HER2 
    # /graphics/scratch3/staff/hosseinza/MIST/PR
    gen_examples("/graphics/scratch3/staff/hosseinza/MIST/PR", num_samples_to_use=4096),
    features=Features(
        he_image=ImageFeature(),
        ihc_image=ImageFeature()
    ),
)
