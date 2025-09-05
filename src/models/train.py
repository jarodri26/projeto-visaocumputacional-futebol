import os, torch, mlflow
from torch import nn, optim
from torch.utils.data import DataLoader
from src.data.dataloader import FramesDataset
from src.models.architectures import get_resnet50
from src.config import load_config
import warnings
warnings.filterwarnings('ignore')

def train_from_folder(frames_dir, out_dir, epochs=1, batch_size=8, lr=1e-4):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = FramesDataset(frames_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = get_resnet50(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    experiment_name = "/Users/<seu-usuario>/<nome-do-experimento>"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                preds = out.argmax(dim=1)
                total += yb.size(0)
                correct += (preds == yb).sum().item()
            acc = correct / max(1, total)
            mlflow.log_metric('acc_epoch', acc, step=epoch)
            print(f'Epoch {epoch} acc={acc:.4f}')
        model_path = os.path.join(out_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        try:
            mlflow.pytorch.log_model(model, 'model')
        except Exception as e:
            print('mlflow logging failed:', e)
        print('Modelo salvo em', model_path)
        return model_path

def evaluate_folder(model_dir, frames_dir):
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataloader import FramesDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet50(num_classes=2)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    model = model.to(device).eval()
    ds = FramesDataset(frames_dir)
    dl = DataLoader(ds, batch_size=8)
    total = 0; correct=0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            total += yb.size(0)
            correct += (preds==yb).sum().item()
    print('Eval acc:', correct / max(1, total))


if __name__ == '__main__':
    # CLI wrapper that loads defaults from configs/config.yaml
    cfg = load_config()
    frames_dir = cfg.get('frames_dir')
    model_dir = cfg.get('model_dir')
    if not frames_dir or not model_dir:
        print('frames_dir or model_dir not configured. Check configs/config.yaml or environment variables.')
    else:
        train_from_folder(frames_dir, model_dir, epochs=1)
