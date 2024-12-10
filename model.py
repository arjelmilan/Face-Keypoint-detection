import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5



class FaceKeypointModel(nn.Module):
  def __init__(self,pretrained = False, requires_grad = True):
    super().__init__()
    if pretrained:
      self.model = shufflenet_v2_x0_5(weights = 'DEFAULT')
    else:
      self.model = shufflenet_v2_x0_5(weights = None)

    if requires_grad:
      for p in self.model.parameters():
        p.requires_grad = True
      print("Training intermediate layer parameters")
    else:
      for p in self.model.parameters():
        p.requires_grad = False
      print("Freezing intermediate layer parameters")

    self.model.fc = nn.Linear(in_features = 1024 , out_features = 136)


  def forward(self,x):
    return self.model(x)

