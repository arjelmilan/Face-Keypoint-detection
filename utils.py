import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib

def dataset_keypoints_plot(data):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        sample = data[i]
        img = sample['image']
        img = np.array(img).astype('float32')
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1],'r.')
    plt.show()
    plt.close()
    
def save_model(epochs, model, optimizer, criterion):
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"outputs/model.pth")