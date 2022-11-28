import torch
from configs import TAGS
from PIL import Image
from torchvision import transforms
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

PATH_MODEL = 'weights/mobilenet_11classes.pt'
DEVICE = 'cpu'

# %%capture
model = torch.load(PATH_MODEL, map_location=torch.device('cpu'))
model.to(DEVICE)

def infer(model, path: str, tags: list):
    img = Image.open(path)
    img = img.resize((224, 224))
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(img).to(DEVICE)
    batch_preds =model(img_tensor.unsqueeze(0))
    idx = torch.argmax(torch.softmax(batch_preds, dim=1), dim=1).cpu().numpy().tolist()[0]
    return tags[idx]


def print_color(image, text):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    (h,w,c)     = image.shape
    org         = (w//2-10,h//2-10)
    fontScale   = 1
    color       = (255, 0, 0)
    thickness   = 1
    
    image_new = cv2.putText(image, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return(image_new)


if __name__ == "__main__": 

    os.makedirs('test_image/', exist_ok = True)
    link = 'C:/Users/This/Desktop/Download/test/'

    files   = os.listdir('C:/Users/This/Desktop/Download/test/')
    for path in files:
        image           = cv2.imread(link + path)
        path_x = path
        path = link + path
        color = infer(model, path, TAGS)
        text = color
        image_new = print_color(image, text)
        cv2.imwrite('test_image/'+ path_x, image_new)


