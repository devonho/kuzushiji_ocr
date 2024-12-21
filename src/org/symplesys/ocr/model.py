from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import Linear

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = Linear(2048,10)
