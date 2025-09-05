from PIL import Image
import torch, os
from torchvision import transforms
from src.models.architectures import get_resnet50

def infer_image(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet50(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    img = Image.open(image_path).convert('RGB')
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    preds = out.argmax(dim=1).item()
    return {'pred': int(preds)}
