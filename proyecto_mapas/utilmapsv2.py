import torch,sys,warnings
import numpy as np
from torch.utils.data import  Dataset, DataLoader
from torchvision.transforms import transforms
warnings.filterwarnings('ignore')
from scipy import signal
import matplotlib.pyplot as plt

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


class MapDataset(Dataset):

    def _extractWindow(self, coordinate):
        image_f, image_c = self.original_image.shape
        padded_image = np.pad(self.original_image[2:-2,2:-2], int(self.window_size/2+3),mode='reflect')

        coord_f = int(coordinate[0] + self.window_size/2+1)
        coord_c = int(coordinate[1] + self.window_size/2+1)

        rvalue = padded_image[coord_f-int(self.window_size/2):coord_f+int(self.window_size/2),coord_c-int(self.window_size/2):coord_c+int(self.window_size/2)]
        return rvalue

    def __init__(self, map_builder, image, image_mask,model_input_size):    
        self.model_input_size = model_input_size
        self.original_image = image
        self.window_size = map_builder.window_size
        self.overlap = map_builder.overlap
        
        targets = np.zeros(image_mask.shape).astype(bool)
        targets[0:: self.window_size-self.overlap,0:: self.window_size-self.overlap] = True
        targets = image_mask & targets
        
        pos = np.where(targets)
        self.coordinates = list(zip(pos[0],pos[1]))
        
    def __len__(self):
        return len(self.coordinates)
        
    def __getitem__(self, idx):
        precise_sample = self._extractWindow(self.coordinates[idx])
        precise_sample = np.stack((precise_sample,precise_sample,precise_sample),axis=2)

        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.model_input_size,self.model_input_size)),
            ])
        return (t(precise_sample).float(), self.coordinates[idx])
        

class MapBuilder:
    def __init__(self, model, window_size=64, overlap=60, batch_size=4000, num_workers=4, model_input_size=224,kern_proportion=None):
        self.model = model
        self.overlap = overlap
        self.model_input_size = model_input_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = 3
        self.kern_proportion = kern_proportion
        
    def _epochProcess(self, loader):
        running_targets = []
        running_coords = []
        
        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        with torch.set_grad_enabled(False):
            for inp, coords in loader:
                inp = inp.to(device)
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = torch.sigmoid(self.model(inp))

                rows = list(coords[0].numpy())
                cols = list(coords[1].numpy())
                               
                running_coords += list(zip(rows,cols))

                preds = output.cpu().detach().numpy()
                running_targets += list(preds)
        return (running_targets, running_coords)
        
    def _votingStrategy(self,coords, image_mask, labels=None):
        np.seterr(divide='ignore', invalid='ignore')
        shape = image_mask.shape
        rmap = np.zeros((shape[0] + self.window_size*2,shape[1] + self.window_size*2))
        m = int(self.window_size/2)
        
        for i in range(len(coords)):
            row,col = coords[i]
            
            if self.kern_proportion is not None:
                if labels is None:
                    rmap[row+self.window_size-m:row+self.window_size+m, col+self.window_size-m:self.window_size+col+m] += 1 * gkern(self.window_size,self.window_size/self.kern_proportion)
                else:
                    rmap[self.window_size+row-m:row+self.window_size+m, col+self.window_size-m:col+self.window_size+m] += labels[i] * gkern(self.window_size,self.window_size/self.kern_proportion)
            else:
                if labels is None:
                    rmap[row+self.window_size-m:row+self.window_size+m, col+self.window_size-m:self.window_size+col+m] += 1 
                else:
                    rmap[self.window_size+row-m:row+self.window_size+m, col+self.window_size-m:col+self.window_size+m] += labels[i]    
                
        rmap = rmap[self.window_size:-self.window_size,self.window_size:-self.window_size]
        return rmap 
    
    def generateMap(self, image, image_mask):
        dataset_from_image = MapDataset(self,image, image_mask,self.model_input_size)
        dataloader_from_image = DataLoader(dataset_from_image, 
                                           batch_size=self.batch_size, 
                                           num_workers=self.num_workers, 
                                           pin_memory = False)
        
        success = False
        old_batch_size = self.batch_size
        while not success:
            try:
                (labels, coords) = self._epochProcess(dataloader_from_image)
                success = True
            except RuntimeError as e:
                self.batch_size = int(self.batch_size*0.5)
                dataloader_from_image = DataLoader(dataset_from_image, 
                                   batch_size=self.batch_size, 
                                   num_workers=self.num_workers, 
                                   pin_memory = False)
                torch.cuda.empty_cache()
                print('New batch size: {}'.format(self.batch_size))
                sys.stdout.flush()

        if old_batch_size != self.batch_size:
            self.batch_size = old_batch_size
            print('Restoring original batch size...')
            print('New batch size: {}'.format(self.batch_size))
            sys.stdout.flush()
        normalization_map = self._votingStrategy(coords,image_mask)
        prevoto = self._votingStrategy(coords,image_mask, labels= [ np.argmax(x) == 1 for x in labels])
        
        result = prevoto /normalization_map
        return result, (normalization_map - normalization_map.min()) / (normalization_map.max() - normalization_map.min()), (prevoto - prevoto.min()) / (prevoto.max() - prevoto.min())
