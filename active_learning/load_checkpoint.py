import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import torchvision.transforms as T

def load_model(checkpoint_path):
    #torch.set_num_threads(1)
    torch.set_grad_enabled(False);
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False, #########
                           num_classes=1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                          strict=False)
    return model

def get_transform():
    transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform