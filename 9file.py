import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_dir = "data/train/"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=35, shuffle=True)

print(len(dataloader.dataset))

test_dir = "data/test/"
tdataset = datasets.ImageFolder(root=data_dir, transform=transform)
tdataloader = torch.utils.data.DataLoader(dataset, batch_size=35, shuffle=True)

model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model.classifier[4] = nn.Linear(4096,1024)
model.classifier[6] = nn.Linear(1024, 1)
print(model.eval())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

runningLoss  = 0
correctPreds = 0
for epoch in range(10):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(images)
        labels = labels.reshape((35, 1)).to(torch.float32)
        o = F.sigmoid(outputs)
        loss = criterion(o, labels)
        
        print(f"Epoch: {epoch}", loss.item())
        runningLoss += loss.item()
                
        loss.backward()
        optimizer.step()

print("LOSS:", runningLoss/len(dataloader.dataset))

torch.save(model.state_dict(), "modelAlex.pth")