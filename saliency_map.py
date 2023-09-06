# CAM visulisation
from torchcam.methods import SmoothGradCAMpp
import baseline
import torch
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import LayerCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import cm
from PIL import Image

import Constant
import data_loader

# def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> Image.Image:
#     """Overlay a colormapped mask on a background image

#     Example::
#         >>> from PIL import Image
#         >>> import matplotlib.pyplot as plt
#         >>> from torchcam.utils import overlay_mask
#         >>> img = ...
#         >>> cam = ...
#         >>> overlay = overlay_mask(img, cam)

#     Args:
#         img: background image
#         mask: mask to be overlayed in grayscale
#         colormap: colormap to be applied on the mask
#         alpha: transparency of the background image
#     """

#     if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
#         raise TypeError('img and mask arguments need to be PIL.Image')

#     if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
#         raise ValueError('alpha argument is expected to be of type float between 0 and 1')

#     cmap = cm.get_cmap(colormap)
#     # Resize mask and apply colormap
#     overlay = mask.resize(img.size, resample=Image.BICUBIC)
#     overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
#     print(overlay[0].shape)
#     mask = overlay > 0
#     # Overlay the image with the mask
#     overlayed_img = Image.fromarray(( np.asarray(img) * mask).astype(np.uint8))

#     return overlayed_img


batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val,_ = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}

GPU = Constant.GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

def check_accuracy(data_loader, model):
        model.eval()
        total_count = 0
        corrects = 0
        for i, data in enumerate(data_loader):

            model.eval()
            inputs = data[0]
            labels = data[1]
            total_count += labels.size(0)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # correct_prediction index
            correct_classifications = (preds == labels)
            correct_pred_samples = inputs[correct_classifications, :, :, :]
            correct_classified_labels = labels[correct_classifications]
            corrects  += correct_classifications.sum()

            j = 0
            print(i)
            


            for each_label in correct_classified_labels:
                if not os.path.exists("./cam/" + str(each_label.item())):
                    os.makedirs("./cam/" + str(each_label.item()))
                save_image(correct_pred_samples[j].cpu().float(), './cam/' + str(each_label.item()) + "/{}.png".format(j))
                j+=1
        print("accuracy:", corrects / total_count)
            # # TODO: store

# TODO: Check correctly predict class, then create folder with the name of the label of the image, then save image to that folder.
# Load model with trained weights-------------------------------------------------

print(os.getcwd)
model_name = "densenet"
num_classes = 50
netC, input_size = baseline.initialize_model(model_name, num_classes, use_pretrained=True)

class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2,self).__init__()
        self.features = netC.features
        # self.fc = nn.Linear(7168, 1024)
        self.classifer = netC.classifier

    def forward(self, x):
        f_c = self.features(x)
        out = F.relu(f_c, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = self.classifer(out)
        return out
model = Classifier2()
model.load_state_dict(torch.load('./trained_models/2_C_model_classification_loss_epoch_200.pt', map_location=torch.device('cpu')))

# CAM ------------------------------------------------------------------------------
cam_extractor = LayerCAM(model)

img = read_image("./feathersv1-dataset/images/piciformes/dendrocopos_major/piciformes_dendrocopos_major_00049.jpg")
# img = Image.open("./feathersv1-dataset/images/trogoniformes/priotelus_temnurus/trogoniformes_priotelus_temnurus_00023.jpg")
# transform = transforms.Resize([240, 40])
# img = transform(img)
# convert_tensor = transforms.ToTensor()
# img = convert_tensor(img)
# Preprocess it for your chosen model
# input_tensor = resize(img, (240, 40)) / 255.
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# input_tensor = img
# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# print(activation_map)

result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

# check_accuracy(loader_val, model)
