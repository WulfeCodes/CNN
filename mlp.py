import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

sigmoid=lambda x: 1 / (1 + np.e^(-x))
#manually define softmax function

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

class MLP:
    def __init__(self, input_size, hidden_size, output_size,lr):  # building the model
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.lr=lr
        #initializing weights and bias terms, will update in backward()
        self.weights_hl1=torch.rand(input_size, hidden_size)
        self.bias_hl1=torch.zeros(hidden_size)

        self.weights_hl2=torch.rand(hidden_size, output_size)
        self.bias_hl2=torch.zeros(output_size)

        """Building hidden layers:

        1st layer:
        [batch_size, input_features]*[input_features, hidden_size(number of neurons)]=[batch_size, hidden_size](weighted_features)
        weighted_features+ [batch_size, hidden_size](bias)=[batch_size, hidden_size](output matrix)
        sigmoid(output_matrix)=activated_output_matrix

        2nd layer:
        activated_output_matrix*[hidden_size, output_size]=[batch_size, output_size]+bias= output_matrix_2
        softmax(output_matrix_2)=[batch_size,output_size] (prediction probabilities) -should be [128,10]
        """
    
    def forward(self, x):  # forward propagation to get predictions
        #assuming x is a tensor with dimensions (128,724)
        weighted_input_hl1=torch.matmul(x, self.weights_hl1)
        output_hl1=weighted_input_hl1 + self.bias_hl1
        activated_output_hl1=output_hl1.apply(sigmoid)
        #activated_output should have dimensions [batch_size, hidden_size]

        weighted_input_hl2=torch.matmul(activated_output_hl1, self.weights_hl2)
        output_hl2=weighted_input_hl2 + self.bias_hl2
        #computing softmax along the rows of hl2, will implement manually later
        outputs=torch.nn.functional.softmax(output_hl2, dim=1)
        return outputs
    
    def backward(self, x, y, pred):
        print(f'Dimensions of pred should be (128,10) {pred.size()}') 
        print(f'Dimensions of y should be (128) atp {y.size()}')
        print(f'Dimensions of x should be (128, 724) {x.size()}')
        # one-hot encode the labels
        encoded_y=torch.zeros(128,10)
        for i in range(y.size(dim=0)):
            encoded_y(i, y(i))=1
        # compute the gradients
        
        # update the weights and biases
        

    def train(self, x,y):
        print('in progress')
        loss=0
        # call forward function
        predictions=self.forward(x)
        # calculate loss
        encoded_y=torch.zeros_like(y)
        for i in range(y.size(dim=0)):
            encoded_y(i, y(i))=1
        #encoded_y and y are both tensors with dimensions (128,10)
        for i in range(y.size(dim=0)):
            loss += y[i,:]*torch.log(encoded_y[i,:]) + (1-y[i,:])*torch.log(1-encoded_y[i,:])
        loss /= (-1/y.size(dim=0))
        # call backward function

        return loss

def main():
    #Define instance of MLP globally so that it's training data (weights and biases) persists when using it on testing data
    
    # First, load data
    train_loader, test_loader = load_data()
    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    hidden_size= 15
    output_size=10
    learning_rate=.2
    model=MLP(input_size, hidden_size, output_size, learning_rate)
    num_epochs = 100
    batch_size=128
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            print(f'Dimensions of inputs: {inputs.size()}, dimensions of lables: {labels.size()}')
            print(labels)

            total_loss += model.train(inputs, labels)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":  # Program entry
    main()  