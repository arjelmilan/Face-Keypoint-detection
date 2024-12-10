
import torch
import tqdm as tqdm

def train_step(model,dataloader, loss_fn,optimizer,device):
  model.train()
  train_loss = 0.0
  for i,data in enumerate(dataloader):
    image , keypoints = data['image'].to(device) , data['keypoints'].to(device)
    keypoints = keypoints.view(keypoints.shape[0],-1)
    output = model(image)
    loss = loss_fn(output , keypoints)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(dataloader)
  return train_loss

def valid_step(model,dataloader,loss_fn,device):
  model.eval()
  valid_loss = 0.0
  with torch.inference_mode():
    for i,data in enumerate(dataloader):
      image , keypoints = data['image'].to(device) , data['keypoints'].to(device)
      keypoints = keypoints.view(keypoints.shape[0],-1)
      output = model(image)
      loss = loss_fn(output , keypoints)
      valid_loss += loss.item()

  valid_loss /= len(dataloader)
  return valid_loss

def train(model , epochs , train_dataloader , valid_dataloader ,optimizer , loss_fn , device):
  result = {
      'train_loss' : [],
      'valid_loss' : []
  }
  for epoch in tqdm(range(epochs)):
    train_loss = train_step(model , train_dataloader , loss_fn , optimizer , device)
    valid_loss = valid_step(model , valid_dataloader , loss_fn , device)
    result['train_loss'].append(train_loss)
    result['valid_loss'].append(valid_loss)
    print(f"Epoch: {epoch} | Train Loss: {train_loss} | Valid Loss: {valid_loss}")

  return result