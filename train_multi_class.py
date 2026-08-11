#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
# the above code is to improve the test code for binary segmentation
# importing libraries####################33
import torch
# Data handling
import pandas as pd
import numpy as np
from losses import*
from utils import*
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
# Path
from pathlib import Path
import time
# tqdm
from tqdm.auto import tqdm
# Torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import os
from utils import*
####################### 1st approach##############33
def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, DEVICE):
    
    model.train()
    
    train_loss = 0.0
    train_accuracy = 0.0
    criterion2 = DiceLoss()
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device = DEVICE, dtype = torch.float32)
        y = y.to(device = DEVICE, dtype = torch.long)
        #print('y',y.shape)
        optimizer.zero_grad()
        logit_mask = model(X)
        #logit_mask=model(X)['out'] # only for advancesegmentationmodel
        #logit_mask= torch.argmax(logit_mask, 1)
        #print('logit_mask ',logit_mask.shape)
        loss = loss_fn(logit_mask, y.squeeze())+criterion2(logit_mask, y.squeeze())
        #loss = loss_fn(logit_mask, y.unsqueeze(dim=1))+criterion2(logit_mask, y.unsqueeze(dim=1))
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        prob_mask = logit_mask.softmax(dim = 1)
        pred_mask = prob_mask.argmax(dim = 1)
        
        tp,fp,fn,tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                            target = y.squeeze().cpu().long(), 
                                            mode = "multiclass", 
                                            num_classes = 5)
        
        train_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
        
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy
##############################for my proposed model#######################
def train_step1(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, 
               loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, DEVICE):
    
    model.train()
    
    train_loss = 0.0
    train_accuracy = 0.0
    criterion2 = DiceLoss()
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device = DEVICE, dtype = torch.float32)
        y = y.to(device = DEVICE, dtype = torch.long)
        optimizer.zero_grad()
        
        # logit_mask = model(X)
        # loss = loss_fn(logit_mask, y.squeeze())
        # train_loss += loss.item()
        # logit_mask = model(X)
        # loss = loss_fn(logit_mask, y.squeeze())
        # val_loss += loss.item()
        ################ my new addition for my dps model######
        logit_mask_1,logit_mask_2,logit_mask_3,logit_mask_4 = model(X)
        
        # ---- loss function ----
        loss4 = loss_fn(logit_mask_4, y.squeeze())
        loss3 = loss_fn(logit_mask_3, y.squeeze())
        loss2 = loss_fn(logit_mask_2, y.squeeze())
        loss1 = loss_fn(logit_mask_1, y.squeeze())
        cri4 = criterion2(logit_mask_4, y.squeeze())
        cri3 = criterion2(logit_mask_3, y.squeeze()) 
        cri2 = criterion2(logit_mask_2, y.squeeze()) 
        cri1 = criterion2(logit_mask_1, y.squeeze())         
        
        loss = loss1+loss2+loss3+loss4+cri1+cri2+cri3+cri4
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        prob_mask = logit_mask_1.softmax(dim = 1)
        pred_mask = prob_mask.argmax(dim = 1)
        
        tp,fp,fn,tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                            target = y.squeeze().cpu().long(), 
                                            mode = "multiclass", 
                                            num_classes = 5)
        
        train_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
        
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)
    
    return train_loss, train_accuracy
####################################################################################################################
def val_step(model:torch.nn.Module, 
             dataloader:torch.utils.data.DataLoader, 
             loss_fn:torch.nn.Module, DEVICE):
    
    model.eval()
    
    val_loss = 0.
    val_accuracy = 0.
    criterion2=DiceLoss()
    ############################################end######
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X = X.to(device = DEVICE, dtype = torch.float32)
            y = y.to(device = DEVICE, dtype = torch.long)
            # print('image',X.shape)
            # print('y',y.shape)
            logit_mask = model(X)
            #logit_mask=model(X)['out'] # only for advancesegmentationmodel
            #print('prd',logit_mask.shape)
            if logit_mask.shape == (1,8,512, 512):
                print(f"Skipping bad label with shape {logit_mask.shape}")
                continue
            loss = loss_fn(logit_mask, y.squeeze())+criterion2(logit_mask, y.squeeze())
            val_loss += loss.item()#
            
            
            ##################### end###############

            prob_mask = logit_mask.softmax(dim = 1)
            pred_mask = prob_mask.argmax(dim = 1)
            
            tp, fp, fn, tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                                   target = y.squeeze().cpu().long(), 
                                                   mode = "multiclass", 
                                                   num_classes = 5)
            
            val_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
            
    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader)
    
    return val_loss, val_accuracy
###############################for my model###############################
def val_step1(model:torch.nn.Module, 
             dataloader:torch.utils.data.DataLoader, 
             loss_fn:torch.nn.Module, DEVICE):
    
    model.eval()
    
    val_loss = 0.
    val_accuracy = 0.
    criterion2=DiceLoss()
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X = X.to(device = DEVICE, dtype = torch.float32)
            y = y.to(device = DEVICE, dtype = torch.long)
            
            # logit_mask = model(X)
            # loss = loss_fn(logit_mask, y.squeeze())
            # val_loss += loss.item()
             ################ my new addition for my dps model######
            logit_mask_1,logit_mask_2,logit_mask_3,logit_mask_4 = model(X)
            
            # ---- loss function ----
            loss4 = loss_fn(logit_mask_4, y.squeeze())
            loss3 = loss_fn(logit_mask_3, y.squeeze())
            loss2 = loss_fn(logit_mask_2, y.squeeze())
            loss1 = loss_fn(logit_mask_1, y.squeeze())
            cri4 = criterion2(logit_mask_4, y.squeeze())
            cri3 = criterion2(logit_mask_3, y.squeeze()) 
            cri2 = criterion2(logit_mask_2, y.squeeze()) 
            cri1 = criterion2(logit_mask_1, y.squeeze())         
            
            loss = loss1+loss2+loss3+loss4+cri1+cri2+cri3+cri4
            #loss = loss_fn(logit_mask, y.squeeze())
            val_loss += loss.item()#
            prob_mask = logit_mask_1.softmax(dim = 1)
            pred_mask = prob_mask.argmax(dim = 1)
            
            tp, fp, fn, tn = smp.metrics.get_stats(output = pred_mask.detach().cpu().long(), 
                                                   target = y.squeeze().cpu().long(), 
                                                   mode = "multiclass", 
                                                   num_classes = 5)
            
            val_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction = "micro").numpy()
            
    val_loss = val_loss / len(dataloader)
    val_accuracy = val_accuracy / len(dataloader)
    
    return val_loss, val_accuracy


####################################################################################################################

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, 
          val_dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, 
          optimizer:torch.optim.Optimizer, epochs:int = 10):
    
    results = {'train_loss':[], 'train_accuracy':[], 'val_loss':[], 'val_accuracy':[]}
    #DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_valid_loss = float("inf")
    checkpoint_path = "/home/muhammad/BCSS-MULTI-Class/pthfiles/NewModelMBLV2Tiger3cls.pth"
    for epoch in tqdm(range(epochs)):
        train_loss, train_accuracy = train_step(model = model, 
                                           dataloader = train_dataloader, 
                                           loss_fn = loss_fn, 
                                           optimizer = optimizer,DEVICE=DEVICE)
        
        val_loss, val_accuracy = val_step(model = model, 
                                     dataloader = val_dataloader, 
                                     loss_fn = loss_fn,DEVICE=DEVICE)
        
        print(f'Epoch: {epoch + 1} | ', 
              f'Train Loss: {train_loss:.4f} | ', 
              f'Train Accuracy: {train_accuracy:.4f} | ', 
              f'Val Loss: {val_loss:.4f} | ', 
              f'Val Accuracy: {val_accuracy:.4f}')
        
        """ Saving the model """
        if val_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
        #EarlyStopping(val_loss, model)
        
        #if EarlyStopping. == True:
            # print("Early Stopping!!!")
            # break
            
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['val_loss'].append(val_loss)
        results['val_accuracy'].append(val_accuracy)
        
    return results

# We define a function to visualize the evolution of the loss and the metric.
def loss_and_metric_plot(results:dict):
    
    training_loss = results['train_loss']
    training_metric = results['train_accuracy']
    
    validation_loss = results['val_loss']
    validation_metric = results['val_accuracy']
    
    fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,3.8))
    ax = ax.flat
    
    ax[0].plot(training_loss, label = "Train")
    ax[0].plot(validation_loss, label = "Val")
    ax[0].set_title("CrossEntropyLoss", fontsize = 12, fontweight = "bold", color = "black")
    ax[0].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[0].set_ylabel("loss", fontsize = 10, fontweight = "bold", color = "black")
    
    ax[1].plot(training_metric, label = "Train")
    ax[1].plot(validation_metric, label = "Val")
    ax[1].set_title("Accuracy", fontsize = 12, fontweight = "bold", color = "black")
    ax[1].set_xlabel("Epoch", fontsize = 10, fontweight = "bold", color = "black")
    ax[1].set_ylabel("score", fontsize = 10, fontweight = "bold", color = "black")
    
    fig.tight_layout()
    fig.show()
    fig.savefig("NewModelMBLV2Tiger625.png")

    #######################3A code hint for 5-fold training
    import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import numpy as np

def train_kfold(model, dataset, loss_fn, optimizer, epochs=10, k=5):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold+1}/{k}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
        
        best_valid_loss = float("inf")
        for epoch in range(epochs):
            train_loss, train_accuracy = train_step1(model, train_loader, loss_fn, optimizer, DEVICE)
            val_loss, val_accuracy = val_step1(model, val_loader, loss_fn, DEVICE)
            
            print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_accuracy:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_accuracy:.4f}')
            
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(model.state_dict(), f'model_fold{fold+1}.pth')
            
            results['train_loss'].append(train_loss)
            results['train_accuracy'].append(train_accuracy)
            results['val_loss'].append(val_loss)
            results['val_accuracy'].append(val_accuracy)
    
    print("Cross-validation completed!")
    return results
  ###############################################################################
#https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114
# the above code is to improve the test code for binary segmentation
# importing libraries####################33
import torch
# Data handling
import pandas as pd
import numpy as np
from utils import save_checkpoint
from utils import EarlyStopping
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
# Path
from pathlib import Path
import time
# tqdm
from tqdm.auto import tqdm
# Torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import os
from model import*
import albumentations as A
from trainutilsapproach1 import*
from trainutilsapproach2 import*

from albumentations.pytorch import ToTensorV2
from dataset import*
from torch.utils.data import DataLoader
from utils import*
import matplotlib.pyplot as plt
##############main function to run the code###############
# for first approach#######
# Training!!!
if __name__ == '__main__':

    
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    EPOCHS = 100
    early_stopping =True
    BATCH_SIZE = 6
    NUM_WORKERS = os.cpu_count()
    ###################################
    # CUDA on using window and linux
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ########for using macbook gpu
    #DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ###################3defining some paramaters for first approach#########33
    #CECKPOINT_SAVE_PATH ="Path to save checkpoint"
    loss_fn = nn.CrossEntropyLoss()
    
    ##################################
    ##################### Working on Pythorch Segmentation Models##################
    # preprocess_input = smp.encoders.get_preprocessing_fn(encoder_name = "resnet34",pretrained = "imagenet")
    # # Define model
    # model = smp.Unet(in_channels = 3, classes = 5)

    # ##**Because we are going to use transfer learning we are going to freeze the encoder layer.**
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    #model = UNet(3,5)
    ####################################################################################
    #model = mobileunetdpslcaim().to(device)
    # model = smp.Unet(
    # encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # for param in model.encoder.parameters():
    #     param.requires_grad = False                   # model output channels (number of classes in your dataset))
    # model = smp.Unet(
    # encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="resnext50_32x4d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="timm-resnest50d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="timm-res2net50_26w_4s",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="timm-regnetx_040",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="timm-gernet_l",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="vgg19",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="timm-mobilenetv3_large_100",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="mit_b2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    # model = smp.Unet(
    # encoder_name="mobileone_s3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    # classes=5, )
    #model = mobileunetdpslcaim(classes=5)
    #model = mobileunetptrlcaim(classes=5)
    #model = efficientunetptrlcaim(classes=5)
    #model = NewModelMBLV31(classes=5)
    #model = NewModelMBLV2DPSwtmsf(classes=5)
    model = NewModelMBLV2(classes=3)
    # from breast_segmentation_train import AdvancedSegmentationModel
    # model = AdvancedSegmentationModel(num_classes=8, dropout_rate=0.3, dropblock_rate=0.1, 
    #               stochastic_depth_rate=0.1, use_attention=False)
    ###################################
    #home/muhammad/BCSS-MULTI-Class/TIGERDATASET/TIGER256overlap128
    TRAIN_IMAGE_PATH = '/home/muhammad/BCSS-MULTI-Class/TIGERDATASET/TIGER512overlap128/train/images/'
    VAL_IMAGE_PATH = '/home/muhammad/BCSS-MULTI-Class/TIGERDATASET/TIGER512overlap128/val/images/'
    TRAIN_MASK_PATH = '/home/muhammad/BCSS-MULTI-Class/TIGERDATASET/TIGER512overlap128/train/masks/'
    VAL_MASK_PATH = '/home/muhammad/BCSS-MULTI-Class/TIGERDATASET/TIGER512overlap128/val/masks/'

    # Define transformations using Albumentations
    transforms_train = A.Compose([A.RandomRotate90(p=0.5),
        
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),ToTensorV2(),
    ])

    transforms_val = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    selected_classes = [0,1, 2]
    train_image_paths, train_mask_paths = load_paths(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH)
    val_image_paths, val_mask_paths = load_paths(VAL_IMAGE_PATH, VAL_MASK_PATH)
    # Create dataset instances
    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, selected_classes=selected_classes,transform=transforms_train)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths,selected_classes=selected_classes, transform=transforms_val)

    print(f'Train Sample: {len(train_dataset)}')
    print(f'Validation Sample: {len(val_dataset)}')



    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # print(len(val_dataloader))
    # print(len(train_dataloader))
    ##################    ###################################First approach for training the model    ###################################

    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
    RESULTS = train(model.to(device = DEVICE), train_dataloader, val_dataloader, loss_fn, 
                    optimizer, EPOCHS)


    loss_and_metric_plot(RESULTS)
    
    # Plotting the training and validation losses
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(RESULTS['train_loss'], label='Train Loss')
    # plt.plot(RESULTS['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()

    # # Plotting Training and Validation Accuracy
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(RESULTS['train_acc'], label='Train Accuracy')
    # plt.plot(RESULTS['val_acc'], label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.show()







    # #################    ###################################second approach for training the model    ##############################################2nd approach#################
    # # Set the maximum learning rate for the optimizer
    # max_lr = 1e-3

    # # Define the number of epochs for training the model
    # num_epochs = 100

    # # Set the weight decay for regularization in the optimizer
    # weight_decay = 1e-4

    # # Define the primary loss function with label smoothing to improve generalization
    # # Label smoothing helps to make the model less confident on the training data
    # criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    # # Define the secondary loss function, Dice Loss, useful for handling class imbalance in segmentation tasks
    # criterion2 = DiceLoss().to(DEVICE)

    # # Initialize the optimizer with weight decay for regularization
    # # AdamW is an optimizer with an adaptive learning rate and weight decay fix
    # optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # # Define a learning rate scheduler
    # # OneCycleLR adjusts the learning rate during training for better convergence
    # # It starts with a lower LR, increases it, and then decreases it towards the end
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=num_epochs,
    #                                                 steps_per_epoch=len(train_dataloader))

    # # Train the model using the defined configurations
    # # The 'fit' function trains the model over the specified number of epochs
    # # and returns a history of training metrics like loss and accuracy
    # history = fit(num_epochs, model, train_dataloader, val_dataloader, criterion1, criterion2, optimizer, scheduler)

    

    # # Plotting the training and validation losses
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Plotting the learning rate
    # plt.subplot(1, 2, 2)
    # plt.plot(history['lrs'], label='Learning Rate')
    # plt.title('Learning Rate Curve')
    # plt.xlabel('Steps')
    # plt.ylabel('Learning Rate')
    # plt.legend()

    # plt.show()

    # # Plotting Training and Validation Accuracy
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_acc'], label='Train Accuracy')
    # plt.plot(history['val_acc'], label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # # Plotting Training and Validation mIoU
    # plt.subplot(1, 2, 2)
    # plt.plot(history['train_miou'], label='Train mIoU')
    # plt.plot(history['val_miou'], label='Validation mIoU')
    # plt.title('Training and Validation mIoU')
    # plt.xlabel('Epochs')
    # plt.ylabel('mIoU')
    # plt.legend()

    # plt.show()
