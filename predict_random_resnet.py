import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    model_path = os.path.join(script_dir, 'resnet18_cifar10.pth')

    if not os.path.exists(model_path):
        print(f"❌ Weights file not found : {model_path}")
        exit()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True, num_workers=0)

    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)

    fig = plt.figure(figsize=(15, 6))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        imshow(images[i])
        
        pred_label = classes[predictions[i]]
        true_label = classes[labels[i]]
        color = 'green' if pred_label == true_label else 'red'
        
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()