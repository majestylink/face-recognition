import torch
import torchvision
from torchvision import datasets, models, transforms
import os
print("Line 5")

# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Line 20")

# Load the dataset
data_dir = './new_dataset'
print("Line 24")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
print("Line 27")
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True)
               for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print("Line 31")

# Load a pre-trained model
model = models.resnet18(pretrained=True, progress=True)
print("Line 35")

# Customize the model for your classification task
num_classes = len(class_names)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
print("Line 40")

# Set the device for training
device = torch.device("cpu")
print("Line 44")
model = model.to(device)
print("Line 46")

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Line 51")

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    print(num_epochs, 'line 57')
    for inputs, labels in dataloaders['train']:
        print(epoch, "Line 59")
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print(epoch, "Line 70")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Line 74")

# Evaluate the model on the test set
model.eval()
print("Line 78")

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Line 893")

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy}%")
