import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        # import pdb
        # pdb.set_trace()

        # Model 1 parameters
        self.m1_conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.m1_conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.m1_conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        # self.linear1 = nn.Linear(in_features = 32 * 18 * 18, out_features = 10368)
        # self.linear2 = nn.Linear(in_features = 10368, out_features = 5184)
        # self.output = nn.Linear(in_features = 5184, out_features = 1103)
        # self.dropout = nn.Dropout(0.15)
        self.m1_linear1 = nn.Linear(in_features = 32 * 16 * 16, out_features = 4096)
        self.m1_linear2 = nn.Linear(in_features = 4096, out_features = 1024)
        self.m1_linear3 = nn.Linear(in_features = 1024, out_features = 10)
        self.m1_output = nn.Linear(in_features = 10, out_features = 2)
        # self.output = nn.Linear(in_features = 2048, out_features = 1103)
        self.m1_dropout = nn.Dropout(0.15)

        # self.linear1 = nn.Linear(28 * 28, 256)
        # self.linear2 = nn.Linear(256, 128)
        # self.linear3 = nn.Linear(128, 256)
        # self.linear4 = nn.Linear(256, 28 * 28)
        # self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # self.conv3 = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        # self.conv4 = nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # self.conv5 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

        # Model 2 parameters
        self.m2_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.m2_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.m2_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m2_linear1 = nn.Linear(in_features=32 * 16 * 16, out_features=4096)
        self.m2_linear2 = nn.Linear(in_features=4096, out_features=2048)
        self.m2_linear3 = nn.Linear(in_features=2048, out_features=1024)
        self.m2_output = nn.Linear(in_features=1024, out_features=398)
        self.m2_dropout = nn.Dropout(0.15)

        # self.linear1 = nn.Linear(28 * 28, 100)
        # self.output1 = nn.Linear(100, 10)
        # self.conv1 = nn.Conv2d(1, 40, 5)
        # self.conv2 = nn.Conv2d(40, 40, 5)
        # self.linear2 = nn.Linear(40 * 4 * 4, 100)
        # self.linear3 = nn.Linear(100, 100)
        # self.linear4 = nn.Linear(40 * 4 * 4, 1000)
        # self.linear5 = nn.Linear(1000, 1000)
        # self.output2 = nn.Linear(1000, 10)
        # self.dropout = nn.Dropout(0.5)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        # elif mode == 3:
        #     self.forward = self.model_3
        # elif mode == 4:
        #     self.forward = self.model_4
        # elif mode == 5:
        #     self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        X = F.max_pool2d(F.relu(self.m1_conv1(X)), 2)
        X = F.max_pool2d(F.relu(self.m1_conv2(X)), 2)
        X = F.max_pool2d(F.relu(self.m1_conv3(X)), 2)
        # X = F.max_pool2d(F.relu(self.conv4(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = F.sigmoid(self.m1_linear1(X))
        X = self.m1_dropout(X)
        X = F.sigmoid(self.m1_linear2(X))
        X = self.m1_dropout(X)
        X = F.sigmoid(self.m1_linear3(X))
        X = self.m1_dropout(X)
        X = self.m1_output(X)
        #
        # X = F.softmax(self.output(X), dim=0)
        # X = X.view(-1, 28 * 28)
        # X = F.relu(self.linear1(X))
        # X = F.relu(self.linear2(X))
        # X = F.relu(self.linear3(X))
        # X = F.relu(self.linear4(X))
        # X = X.view(-1, 1, 28, 28)
        # 
        # X = X.view(-1, self.num_flat_features(X))
        # X = F.sigmoid(self.linear1(X))
        # X = F.sigmoid(self.output1(X))
        #
        # Uncomment the following return stmt once method implementation is done.
        return  X
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        X = F.max_pool2d(F.relu(self.m2_conv1(X)), 2)
        X = F.max_pool2d(F.relu(self.m2_conv2(X)), 2)
        X = F.max_pool2d(F.relu(self.m2_conv3(X)), 2)
        # X = F.max_pool2d(F.relu(self.conv4(X)), 2)
        X = X.view(-1, self.num_flat_features(X))
        X = F.relu(self.m2_linear1(X))
        X = self.m2_dropout(X)
        X = F.relu(self.m2_linear2(X))
        X = self.m2_dropout(X)
        X = F.sigmoid(self.m2_linear3(X))
        X = self.m2_dropout(X)
        X = self.m2_output(X)
        #
        # X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        # X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        # X = F.upsample_bilinear(F.relu(self.conv3(X)), scale_factor = 2)
        # X = F.upsample_bilinear(F.relu(self.conv4(X)), scale_factor = 2)
        # X = F.sigmoid(self.conv5(X))
        # 
        # X = F.max_pool2d(F.sigmoid(self.conv1(X)), 2)
        # X = F.max_pool2d(F.sigmoid(self.conv2(X)), 2)
        # X = X.view(-1, self.num_flat_features(X))
        # X = F.sigmoid(self.linear2(X))
        # X = F.sigmoid(self.output1(X))
        #
        # Uncomment the following return stmt once method implementation is done.
        return  X
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # # Replace sigmoid with ReLU.
    # def model_3(self, X):
    #     # ======================================================================
    #     # Two convolutional layers + one fully connected layer, with ReLU.
    #     #
    #     # ----------------- YOUR CODE HERE ----------------------
    #     X = F.max_pool2d(F.relu(self.conv1(X)), 2)
    #     X = F.max_pool2d(F.relu(self.conv2(X)), 2)
    #     X = X.view(-1, self.num_flat_features(X))
    #     X = F.relu(self.linear2(X))
    #     X = F.relu(self.output1(X))
    #     #
    #     # Uncomment the following return stmt once method implementation is done.
    #     return  X
    #     # Delete line return NotImplementedError() once method is implemented.
    #     # return NotImplementedError()

    # # Add one extra fully connected layer.
    # def model_4(self, X):
    #     # ======================================================================
    #     # Two convolutional layers + two fully connected layers, with ReLU.
    #     #
    #     # ----------------- YOUR CODE HERE ----------------------
    #     X = F.max_pool2d(F.relu(self.conv1(X)), 2)
    #     X = F.max_pool2d(F.relu(self.conv2(X)), 2)
    #     X = X.view(-1, self.num_flat_features(X))
    #     X = F.relu(self.linear2(X))
    #     X = F.relu(self.linear3(X))
    #     X = F.relu(self.output1(X))
    #     #
    #     # Uncomment the following return stmt once method implementation is done.
    #     return  X
    #     # Delete line return NotImplementedError() once method is implemented.
    #     # return NotImplementedError()

    # # Use Dropout now.
    # def model_5(self, X):
    #     # ======================================================================
    #     # Two convolutional layers + two fully connected layers, with ReLU.
    #     # and  + Dropout.
    #     #
    #     # ----------------- YOUR CODE HERE ----------------------
    #     X = F.max_pool2d(F.relu(self.conv1(X)), 2)
    #     X = F.max_pool2d(F.relu(self.conv2(X)), 2)
    #     X = X.view(-1, self.num_flat_features(X))
    #     X = F.relu(self.linear4(X))
    #     X = self.dropout(X)
    #     X = F.relu(self.linear5(X))
    #     X = self.dropout(X)
    #     X = F.relu(self.output2(X))
    #     #
    #     # Uncomment the following return stmt once method implementation is done.
    #     return  X
    #     # Delete line return NotImplementedError() once method is implemented.
    #     # return NotImplementedError()
    
    # 
    def num_flat_features(self, X):
        size = X.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features