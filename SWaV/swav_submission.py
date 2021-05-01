import torch
import torch.nn as nn
from torchvision.transforms import transforms
from fastai.vision.models import xresnet

team_id = 15
team_name = "loSSLess"
email_address = "vvb238@nyu.edu"



class SwaVResNet(torch.nn.Module):

    def __init__(self,encoder_path=None):
        super().__init__()
        self.encoder=torch.nn.Sequential(*(list(xresnet.xresnet18(pretrained=False).children()))[:-1])
        if encoder_path:
            checkpoint = torch.load(encoder_path,map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint)
        self.classifier=nn.Linear(in_features=512,out_features=800)
    
    def forward(self,x):
        rep=self.encoder(x).view(x.shape[0],-1)
        y_hat=self.classifier(rep)
        return y_hat


def get_model():
    trained_classifier=SwaVResNet(encoder_path=None)
    trained_classifier.load_state_dict(torch.load('25_resnet18.pth'))
    trained_classifier.eval()
    return trained_classifier

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# model=get_model()
# print(model)
