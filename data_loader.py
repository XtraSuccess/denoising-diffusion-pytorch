import torch
import os
import PIL as pillow
DataUtils = torch.utils.data
Dataloader = DataUtils.DataLoader
import torchvision as TV; 


        
        

class ImgDataSet(DataUtils.Dataset):
    def get_files(self, directories) -> list[str]:
        images = []
        for path in directories:
            files = os.listdir(path)
            
            images.extend([ os.path.abspath(path+"/"+img)
                           for img in files if 
                            img.endswith(".png") 
                            or img.endswith(".jpg")
                            or img.endswith(".jpeg")])
            
        return images
    
    def __init__(self, xsize, ysize, *directories):
        (self.xSize, self.ySize) = xsize, ysize

        self._paths = self.get_files(directories)

        self.len = len(self._paths)

    def transform_img(self, imgTensor: torch.Tensor ):
        img = TV.transforms.Resize(
            (self.xSize, self.ySize), antialias =True)(
            imgTensor
            )
        img = img.type(torch.FloatTensor)
        #print(img.shape, "\n\n")

        img = TV.transforms.RandomHorizontalFlip()(img)
        

        return img/255
        
    def load_img(self, img:str) ->torch.Tensor:
        im = pillow.Image.open(img).convert("RGB")
        return TV.transforms.PILToTensor()(im)
        
    def __len__(self):
        return self.len

    def __getitem__(self,idx) -> torch.Tensor:
        img = self.load_img(self._paths[idx//2])
        
        return self.transform_img( img)
    
    def get_loader(self, batch_size:int ) -> DataUtils.DataLoader:
            return DataUtils.DataLoader(self, batch_size = batch_size, 
                                        shuffle = True, num_workers = 0)