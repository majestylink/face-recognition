import os

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

class_names = ['atiku', 'buhari', 'cristiano_ronaldo', 'kylian_mbappe', 'lionel_messi', 'peter_obi', 'tinubu']

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def process_image(photo_path):
    # Create an instance of the model
    model = models.resnet18(pretrained=False)
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model state dictionary
    print("Loading artifact...")
    model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the photo
    photo = Image.open(photo_path)
    input_tensor = transform(photo)
    input_tensor = input_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, predicted_class = torch.max(output, 1)

    # Get the predicted class label and probability score
    predicted_label = class_names[predicted_class.item()]
    probability_score = probabilities[predicted_class.item()]
    threshold = 0.9
    if probability_score < threshold:
        return "Person not recognized"
    else:
        return {
            'predicted_label': predicted_label,
            'probability_score': probability_score.item()
        }
