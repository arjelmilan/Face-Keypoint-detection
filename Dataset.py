import torch
import cv2
import albumenations as A



train_transform = A.Compose([
    A.Resize(height = 224,
             width = 224,
             always_apply = True),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=35, p=0.5),
    A.RandomGamma(p=0.3),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    ],keypoint_params = A.KeypointParams(format = 'xy' , remove_invisible = False))

valid_transform = A.Compose([
    A.Resize(height = 224,
             width = 224,
             always_apply = True),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    ],keypoint_params = A.KeypointParams(format = 'xy' , remove_invisible = False))


class FaceKeypointDataset(Dataset):
  def __init__(self,path,data,is_train):
    self.path = path
    self.data = data
    self.is_train = is_train

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    image = cv2.imread(os.path.join(self.path,self.data.iloc[idx][0]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    keypoints = self.data.iloc[idx][1:]
    keypoints = np.array (keypoints,dtype = 'int')
    keypoints = keypoints.reshape(-1,2)

    if self.is_train:
      transformed = train_transform(image = image , keypoints = keypoints)
    else:
      transformed = valid_transform(image = image , keypoints = keypoints)

    image = transformed['image']
    keypoints = transformed['keypoints']

    image = np.transpose(image , (2,0,1))

    return {
        'image' : torch.tensor(image,dtype = torch.float),
        'keypoints' : torch.tensor(keypoints,dtype = torch.float)
    }

