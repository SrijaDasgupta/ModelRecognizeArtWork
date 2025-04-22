from __future__ import print_function
import argparse
import os
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from ConvNet import ConvNet
from CustomDataLoader import CustomDataLoader
from BaseCustomDataLoader import BaseCustomDataLoader
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    # import pdb
    # pdb.set_trace()
    # print("haha")
    for batch_idx, batch_sample in enumerate(train_loader):
        pauseTrainHook = glob.glob(os.path.join(os.getcwd(), "hook.txt"))
        if pauseTrainHook:
            import pdb
            pdb.set_trace()
            os.remove(os.path.join(os.getcwd(), "hook.txt"))
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)

        # #testing
        # nTarget = np.zeros((10, 10))
        # # nTarget.to(device)
        # for idx in range(10):
        #     nTarget[idx][target[idx]] = 1
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign correct loss function.
        # loss = NotImplementedError()
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign counting function for correct predictions.
        # correct = NotImplementedError()
        # correct += (pred.T[0] == target).sum().item()
        for entryIdx in range(len(output)):
            if target[entryIdx][pred[entryIdx]] == 1:
                correct += 1
        
    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx+1) * batch_size)
    print('Train Set\nAverage Loss : {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100. * correct / ((batch_idx+1) * batch_size)))
    return train_loss, train_acc
    

def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign correct loss function.
            # Compute loss based on same criterion as training 
            # loss = NotImplementedError()
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # ======================================================================
            # Count correct predictions overall 
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign counting function for correct predictions.
            # correct = NotImplementedError()
            # import pdb
            # pdb.set_trace()
            # correct += (pred.T[0] == target).sum().item()
            for entryIdx in range(len(output)):
                if target[entryIdx][pred[entryIdx]] == 1:
                    correct += 1

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test Set\nAverage Loss : {:.4f}, Accuracy : {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy
    

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # PREVIOUS_MODEL = None
    PREVIOUS_MODEL = os.path.join("TrainedModels", "model_epoch_55.h5")
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch Device Selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode).to(device)
    if PREVIOUS_MODEL:
        model.load_state_dict(torch.load(PREVIOUS_MODEL))
    
    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign correct loss function.
    # criterion = NotImplementedError()
    criterion = nn.MSELoss()
    
    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign appropriate optimizer with learning rate and other paramters.
    # optimizer = NotImplementedError()
    optimizer = optim.Adam(model.parameters(), lr = FLAGS.learning_rate)
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    # dataset1 = datasets.CIFAR10('./data/', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.CIFAR10('./data/', train=False,
    #                    transform=transform)
    # import pdb
    # pdb.set_trace()

    # Full DataSet
    # datasetPath = pd.read_csv('imet-2019-fgvc6/train.csv')
    # dataSet = CustomDataLoader(rootDir = Path("Resize_300"), df = datasetPath, transforms = transform, size=128)
    # trainSize = int(0.2 * len(dataSet))
    # testSize = len(dataSet) - trainSize
    # dataSet, _ = random_split(dataSet, [trainSize, testSize])
    # trainSize = int(0.8 * len(dataSet))
    # testSize = len(dataSet) - trainSize
    # trainDataset, testDataset = random_split(dataSet, [trainSize, testSize])

    baseTruthPath = pd.read_csv(FLAGS.truth_csv)
    # import pdb
    # pdb.set_trace()
    baseDataSet = BaseCustomDataLoader(rootDir = Path("Resize_128"), df = baseTruthPath, transforms = transform, mode=FLAGS.mode)
    trainSize = int(0.8 * len(baseDataSet))
    testSize = len(baseDataSet) - trainSize
    trainDataset, testDataset = random_split(baseDataSet, [trainSize, testSize])
    train_loader = DataLoader(trainDataset, batch_size = FLAGS.batch_size,
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(testDataset, batch_size = FLAGS.batch_size,
                                shuffle=False, num_workers=4)

    
    best_accuracy = 0.0

    trainLosses = []
    trainAccuracies = []
    testLosses = []
    testAccuracies = []
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        startTime = time.time()
        print('\nEpoch : {}'.format(epoch))
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch, FLAGS.batch_size)
        # print('Train Set\nAverage Train Loss : {:.6f}'.format(train_loss))
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        # print('Test Set\nAverage Test Loss : {:.6f}'.format(test_loss))
    
        trainLosses.append(train_loss)
        trainAccuracies.append(train_accuracy)
        testLosses.append(test_loss)
        testAccuracies.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        print("Epoch Time : {}".format(time.time() - startTime))
        torch.save(model.state_dict(),
                   os.path.join("TrainedModels", "model_epoch_{}.h5".format(epoch)))
    
    
    print("Accuracy is {:2.2f}".format(best_accuracy))
    
    print("Training and evaluation finished")

    plt.subplots_adjust(hspace = 0.3)      
    plt.subplot(2, 1, 1).set_title("Train Vs Test Loss")
    plt.plot(trainLosses, label = 'Train Loss')
    plt.plot(testLosses, label = 'Test Loss')
    plt.legend()
    plt.subplot(2, 1, 2).set_title("Train Vs Test Accuracy")
    plt.plot(trainAccuracies, label = 'Train Accuracy')
    plt.plot(testAccuracies, label = 'Test Accuracy')
    plt.legend()
    plt.savefig('Plot_Mode_{}_LR_{}.png'.format(FLAGS.mode, FLAGS.learning_rate), dpi = 1000)

    # plt.title("Train Vs Test Loss")
    # plt.plot(trainLosses, label = 'Train Loss')
    # plt.plot(testLosses, label = 'Test Loss')
    # plt.legend()
    # plt.savefig('Plot_Loss_Mode_{}.png'.format(FLAGS.mode), dpi = 1000)
    # plt.clf()

    # originalImages = []
    # reconstructedImages = []
    # classCounter = [0 for i in range(10)]
    # with torch.no_grad():
    #     for batch_idx, sample in enumerate(test_loader):
    #         data, target = sample
    #         if len(originalImages) < 20:
    #             for image, label in zip(data, target):
    #                 if classCounter[label] < 2:
    #                     classCounter[label] += 1
    #                     originalImages.append(image)
        
    #     originalImages = torch.cat(originalImages).reshape(20, 1, 28, 28)
    #     reconstructedImages = model(originalImages)

    #     plt.figure(figsize = (7, 1))
    #     for i in range(40):
    #         if i < 20:
    #             plt.subplot(2, 20, i + 1)
    #             plt.imshow(originalImages[i].numpy().reshape(28, 28))
    #             plt.gray()
    #             plt.axis(False)
    #         else:
    #             plt.subplot(2, 20, i + 1)
    #             plt.imshow(reconstructedImages[i - 20].numpy().reshape(28, 28))
    #             plt.gray()
    #             plt.axis(False)
    #     plt.savefig('Show_Images_Mode_{}.png'.format(FLAGS.mode), dpi = 1000)
    
    
if __name__ == '__main__':
    # Set parameters for iMet 2019 Kaggle Challenge
    parser = argparse.ArgumentParser('iMet 2019 Kaggle Challenge')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-2.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=128,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--truth_csv',
                        type=str,
                        default='baseTruth.csv',
                        help='Ground Truth file to be read.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)