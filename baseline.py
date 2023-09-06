# baseline DenseNet121

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import Constant

import data_loader
import torch.nn.functional as F



import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

batch_size = 128
# why? it should be 595
num_classes = 595
num_epochs = 100

GPU = Constant.GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

the_data_loader = data_loader.Data_Loader(batch_size)
loader_train, loader_val, loader_test = the_data_loader.loader()
dataloaders = {'train': loader_train,
                'val' :loader_val}


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    patience, optimal_val_loss = 5, np.inf

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                print(i)
                # zero the parameter gradients
                inputs = data[0].to(device)
                labels = data[1].to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    print(outputs)

                    loss = criterion(outputs, labels)
                    print("loss:", loss)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if Constant.colab:
                    torch.save(best_model_wts, "data_augmentation/trained_models/denseNet121.pt")
                else:
                    torch.save(best_model_wts, "./trained_models/denseNet121.pt")
                patience -= 1
                if patience <= 0:
                    print("early stopped")
                    break
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                patience = 5

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def validate(testloader, netC):
  print("validating----------------------")
  netC.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for i, data in enumerate(testloader):
          inputs, labels = data
          inputs, labels = data[0].to(device), data[1].to(device)
          outputs = netC(inputs)
          print(outputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = (correct / float(total)) * 100 

  print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / float(total)))
  netC.train()

# Initialize the model for this run
model_name = "densenet"
model_ft, input_size = initialize_model(model_name, num_classes, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)


model_ft = model_ft.to(device)
print(device)

params_to_update = model_ft.parameters()
# print("Params to learn:")

# for name,param in model_ft.named_parameters():
#     if param.requires_grad == True:
#         print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# Train and evaluate
if __name__ == "__main__":
    trained_model, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)
    validate(loader_test, trained_model)