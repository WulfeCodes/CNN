import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

np.set_printoptions(precision=4, suppress=False)

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
#Numerically stable softmax, axis=0 means that the computation happens column by column
def softmax(matrix_array):
    matrix_array_stable = matrix_array - np.max(matrix_array, axis=0, keepdims=True)
    matrix_array_stable= np.exp(matrix_array_stable)
    matrix_array_stable = matrix_array_stable / np.sum(matrix_array_stable, axis=0, keepdims=True)
    return matrix_array_stable
#numerically stable sigmoid, when there are large negative values must compute e^x/(1+e^x)
def sigmoid(matrix_array):
    return (np.where(matrix_array<0, 
    np.exp(matrix_array)/ (1 + np.exp(matrix_array)),
    1 / (1+np.exp(-matrix_array))
    ))

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):  # building the model
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.lr=lr
        #initializing weights and bias terms, will update in backward()
        rng=np.random.default_rng()
        self.w_1=rng.normal(0, 1, (hidden_size, input_size))
        self.b_1=np.zeros((hidden_size,1), dtype=np.float64)

        self.w_2=rng.normal(0, 1, (output_size, hidden_size))
        self.b_2=np.zeros((output_size,1), dtype=np.float64)

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
        #assuming x is a matrix with dimensions (input_size, batch_size)
        batch_size=x.shape[1]
        z_1=np.matmul(self.w_1, x) + np.matmul(self.b_1, np.ones((1, batch_size)))
        #apply sigmoid function elementwise to z_1 
        a_1 = sigmoid(z_1)
        #a_1 should have dimensions [hidden_size, batch_size]

        z_2=np.matmul(self.w_2, a_1) + np.matmul(self.b_2, np.ones((1, batch_size)))
        #z_2 has dimensions output_size, batch_size, apply softmax function on columns
        #call the vectorized softmax operation on z_2, a_2 has dimensions [output_size, batch_size]
        a_2=softmax(z_2)
        return a_2
    
    def backward(self, x, y, pred):
        print(f'Dimensions of pred should be (128,10) {pred.size()}') 
        print(f'Dimensions of y should be (128) atp {y.size()}')
        print(f'Dimensions of x should be (128, 724) {x.size()}')
        # one-hot encode the labels
        """ 
        encoded_y=torch.zeros(128,10)
        for i in range(y.size(dim=0)):
            encoded_y(i, y(i))=1
            """
        # compute the gradients
        
        # update the weights and biases
        

    def train(self, x,y):
        loss=0
        batch_size=x.shape[1]
        
        y_hat=self.forward(x)
        #One hot encode the vectors for y, then calculate the loss with cross entropy
        #Call backwards update weights and biases within backwards return the loss and then train is called again in the loop
        
        encoded_y=np.zeros((self.output_size, batch_size))
        for i in range(y.shape[0]):
            encoded_y[y[i], i]=1
        #encoded_y and y are both tensors with dimensions (10, 128)
        #cross entropy loss by looping through columns might be able to vectorize
        for i in range(batch_size):
            loss += np.sum(-encoded_y[:,i] * np.log(y_hat[:,i]))
        print(loss)
        return loss

def main():
    #Define instance of MLP globally so that it's training data (weights and biases) persists when using it on testing data
    #Could make input size a tuple with batch size included, could make batch size a global variable? 
    # First, load data
    train_loader, test_loader = load_data()
    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    hidden_size= 15
    output_size=10
    learning_rate=.2
    model=MLP(input_size, hidden_size, output_size, learning_rate)
    num_epochs = 100
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # Loop is for mini batch training 
            x=np.empty((784,128))
            inputs=inputs.numpy()
            for i in range(inputs.shape[0]):
                x[:, i]=np.ravel(inputs[i,...]) 
            labels=labels.numpy()
            
            total_loss += model.train(x, labels)
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