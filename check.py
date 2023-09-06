classifier_path = "/content/gdrive/MyDrive/ic/trained_models/top_50/2_C_model_classification_loss_epoch_1400.pt"

generator_path = "/content/gdrive/MyDrive/ic/trained_models/top_50/2_G_model_classification_loss_epoch_1300.pt"

from CVAEGAN import VAEGAN

import torch

import data_loader

import torch.nn.functional as F


device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")


batch_size = 64
the_data_loader = data_loader.Data_Loader(batch_size)

loader_train, loader_val, loader_test = the_data_loader.loader()

net = VAEGAN(150).cuda()

net.classifier.load_state_dict(torch.load(classifier_path))
net.decoder.load_state_dict(torch.load(generator_path))

accuracy_list = []
for i,data in enumerate(loader_train):
    inputs = data[0].to(device)
    labels = data[1].to(device)

    one_hot_class = F.one_hot(labels, num_classes=50)
    random_z = torch.randn(len(inputs), net.z_size).cuda()
    fake_images = net.decoder(random_z, one_hot_class)

    classification_result_fake, feature_c_fake = net.classifier(fake_images)
    _, preds_fake = torch.max(classification_result_fake, 1)

    correct = (preds_fake == labels).sum().item()
    current_accuracy = (correct / float(labels.size(0)))

    accuracy_list.append(current_accuracy)

print("average accuracy:",  sum(accuracy_list)/ float(len(accuracy_list)))