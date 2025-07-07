import torch
from resnet_cifar10_V2 import ResNet18_CIFAR10_V2

def load_model(checkpoint_path, device):
    model = ResNet18_CIFAR10_V2(num_classes=10)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "resnet_cifar10_V2.pth"
    model = load_model(checkpoint_path, device)
    name, param = next(model.named_parameters())
    print(name, tuple(param.shape))
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        output = model(dummy_input)
        pred = torch.softmax(output, dim=1).argmax(dim=1).item()
    print(pred)

