import kagglehub
import idx2numpy
import os
# Download latest version


import numpy as np


#TODO fix backprop for CNN pooling and filter layers, it should reference a outList
#TODO set y_actual NN2 as parammeter based on dataset label 

class NeuralNetLayer1:
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layerList = np.random.rand(output_dims,input_dims) * np.sqrt(2/input_dims)
        self.outputList = np.ndarray((output_dims,1))
        self.pre_outputList = np.ndarray((output_dims,1))
        self.differentiated_outputList = np.ndarray((output_dims,1))
        # self.init_cached_weightMatrices()
        #TODO call cached weight Matrices but with proper xavier method
        self.input = None
    # def init_cached_weightMatrices(self):
        
    #     for i in range(64):
    #         temp = np.random.randn(1,self.input_dims)
    #         self.layerList[i]=temp
            
    def forward(self,input):
        self.input = input
        # print("hello here are the shapes")
        # print(self.layerList.shape)
        # print(self.outputList.shape)
        # print(input.shape)
        for i in range(64):
            self.outputList[i]=(self.layerList[i]@input)

            for j in range(len(self.outputList[i])):
                self.pre_outputList=self.outputList
                self.outputList[i,j]=self.leaky_relu_inline(self.outputList[i,j])

    def leaky_relu_inline(self,x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    def backPropogateCalc(self,delta2):
        self.losser_vector=np.ndarray((64,1))

        for i in range(64):
            if self.pre_outputList[i]>0:
                self.losser_vector[i]=1
            else:
                self.losser_vector[i]=.01

        #returning delta1 here, delta2 here,

        # self.differentiated_outputList=self.pre_outputList * self.outputList.T
        #size 64x1
        # print("self layer list dimensions:, should be of 64 x input_dims",self.layerList.shape)
        # print("delta dimensions: should be of 64x1",delta2.shape)
        # print("losser vector should be of 64x1 shape", self.losser_vector.shape)

        return self.layerList.T @ (delta2 * self.losser_vector)
        #self.LayerList is 64 x input_dims times delta 2 which is 64x1 equaling our input dim of 1936x1

    def backPropogateWeights(self,delta2):
        #partial of H_w{i}=input_i@deltai+1, for here it should equal 64 x 1936 dims
        #delta2 is 64x1 self.input is 1936x1
        self.layerList -= (0.001 * delta2 @ self.input.T)
        print("updated the weights!")
        

class NeuralNetLayer2:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        #64 10x1 vectors.T, changed from 10,1,64 deleted batch dim
        self.layerList = np.random.rand(output_dims,input_dims) * np.sqrt(2/input_dims)
        self.outputList = np.ndarray((output_dims,1))
        self.pre_outputList = np.ndarray((output_dims,1))
        #temp y_actual, should be passed in a parameter. thatll be known once I analyze the mnist dataset
        self.y_actual = 0
        self.sum = None
        self.y_actual_index = self.y_actual -1
        # self.init_cached_weightMatrices()
        self.jacobian_sigmoid = np.ndarray((output_dims,output_dims))
        self.differentiated_outputVector = np.zeros((output_dims,1))
        self.delta2 = np.zeros((output_dims,1))
        self.delta1 = np.zeros((output_dims,1))
    #not needed anymore because of random.rand constructor 
    # def init_cached_weightMatrices(self):
        
    #     for i in range(10):
    #         temp = np.random.randn(1,self.input_dims)
    #         self.layerList[i] = temp


    def forward(self,input):
        for i in range(10):
            self.outputList[i]=(self.layerList[i]@input)
        self.input = input
    
    def softmaxxer(self):
        #nice, finally a vectorized computation!
        self.pre_outputList=self.outputList
        shifted = self.outputList - np.max(self.outputList)  # Prevent overflow
        exps = np.exp(shifted)
        self.outputList = exps / (np.sum(exps) + 1e-10)  # Add epsilon
        print("softmaxxed output:",self.outputList)

    def calc_loss(self):
        val = self.outputList[self.y_actual_index]
        safe_val = max(val, 1e-10)  # Increase this from 1e-12
        loss = -np.log(safe_val)
        # print("here are the one hot encoded preds and its adjusted val", val, safe_val)
        print("index of pred: ", self.y_actual_index)

        # Add clipping to prevent extreme gradient values
        loss = np.clip(loss, -100, 100)
        print(f"loss: {loss}")
        # Safer derivative calculation
        self.differentiated_outputVector = np.zeros((10,1))
        self.differentiated_outputVector[self.y_actual_index] = np.clip(-1.0 / safe_val, -100, 100)

    def calc_sigmoid_activation_loss(self):
        #derivation of jacobian is given by a(1-a) for diagonal elements
        #-a_ia_j for off diagonal elements
        #try to switch these to vectorized computations
        for i in range(10):
            for j in range(10):
                epsilon = 1e-12  # Safety to avoid log(0), div by 0
                # -- For i --
                shifted_i = self.pre_outputList[i] - np.max(self.pre_outputList[i])
                exp_i = np.exp(shifted_i)
                sum_exp_i = np.sum(exp_i) + epsilon  # Add epsilon for stability
                exp_scores = exp_i / sum_exp_i       # Stable softmax for i

                # -- For j --
                shifted_j = self.pre_outputList[j] - np.max(self.pre_outputList[j])
                exp_j = np.exp(shifted_j)
                sum_exp_j = np.sum(exp_j) + epsilon
                exp_score2 = exp_j / sum_exp_j 
                
                if i==j:
                    #corrected the sigmoid backprops!
                    self.jacobian_sigmoid[i,j]=((exp_scores)*(1-(exp_scores))).item()

                else:
                    self.jacobian_sigmoid[i,j]=-(((exp_scores)*(exp_score2))).item()
        

    
    def backPropogateCalc(self):
        #i dont think there is a negative here, if its just off of cross entropy/expected likelihood
        self.calc_sigmoid_activation_loss()

        softmax_loss_vector = np.ndarray((10,1))

        # for i in range(10):
        #     sigmoid_loss_vector[i] = self.jacobian_sigmoid[i,self.y_actual_index]
        
        for i in range(10):
            if i==self.y_actual_index:
                softmax_loss_vector[i] = self.outputList[i] - 1
            else:

                softmax_loss_vector[i] = self.outputList[i]
    
        self.delta2 = softmax_loss_vector
        print("delta 2 norm: ")
        self.delta2 = clip_norm(self.delta2,100)
        self.delta1 = self.layerList.T @ self.delta2 

        return self.delta1
    
    def backPropogateWeights(self):

    #takes in a 64x1 vector,
    #multiplied by a 10x64 weights, delta2 is 10x1
        self.layerList -= (0.001 * self.delta2 @ self.input.T)
        print("layer weights 2 updated!")
 
class outputFilter1:
    def __init__(self,input_dims, filter_kernel_dims, filter_num , stride):
        self.output_dims = int((np.floor(input_dims-filter_kernel_dims)/stride)+1)
        self.input_dims = input_dims
        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        ##TODO change randomization to be of xavier/fan in variant 
        
        self.filterList = np.random.rand(filter_num,self.filter_kernal_dims,self.filter_kernal_dims) * np.sqrt(2/input_dims)

        self.initKernels()


        #changed this from: 
        # self.OutList = np.zeros((filter_num*prev_layer_filter_num,self.output_dims,self.output_dims))
        self.OutList = np.zeros((filter_num,self.output_dims,self.output_dims))
        self.preOutList = np.zeros((filter_num,self.output_dims,self.output_dims))
        #filterList and output.
        self.temp = np.zeros((self.output_dims,self.output_dims))

    def initKernels(self):
        for i in range(self.filter_num):
            print("ranning!")
            self.filterList[i]= np.random.randn(self.filter_kernal_dims,self.filter_kernal_dims) * np.sqrt(2/self.filter_kernal_dims)
            
    def forward(self,input,filter,k): 
        temp = np.zeros((self.output_dims,self.output_dims))
        pretemp = np.zeros((self.output_dims, self.output_dims))
        # print("temp dimensions", temp.shape)
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #if stride were to take effect, just change the iterator of i and j for loops
        #TODO decouple leaky relu, would be done by done at backpropogation
        # print("output dims equals: ",self.output_dims)
        for i in range(self.output_dims):
            # print("i is ", i)
            for j in range(self.output_dims):
                # print("j is: ",j)
                # print("shape of actual input",input.shape)
                # print("shape of input",input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims].shape)
                # print("shape of filter")
                
                #hadamard product or reg matrix multiplication?s
                pretemp[i][j]=np.sum(input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]*filter)
                temp[i][j]=self.leaky_relu_inline(np.sum(input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]*filter))
        self.preOutList[k]=pretemp
        return temp
    
    def leaky_relu_inline(self,x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    def leaky_relu_derivative(self,x,alpha=0.01):
        return 1 if x > 0 else alpha
    
    def backwardMatrices(self,kernel,losses,index):
        #range of losses equates to stretched dimensions of output_dims w each index being one entry
        temp = np.zeros((self.input_dims,self.input_dims))

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                #differentiated to create delta meaning the left over would be the weight kernal!
                temp[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims] += kernel * (losses[i,j]* self.leaky_relu_derivative(index[i,j]))
                #matrix multiplied with a scalar
        # print("we finished computing temp!", temp)
        return temp
    
    def backPropogateWeights(self,input,lossDelta):
        temp = np.zeros((self.filter_kernal_dims,self.filter_kernal_dims))
        for i in range(self.output_dims):
            for j in range(self.output_dims):
                region = input[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims]
                region_sum = np.sum(region)
                derivative = self.leaky_relu_derivative(region_sum)
                delta_val = lossDelta[i, j] * derivative
                scaled_region = region * delta_val
                temp += 0.001 * scaled_region  
        return temp

class poolingLayer1:
    def __init__(self,input_dims, filter_kernel_dims,filter_num,stride):
        self.output_dims = int((np.floor(input_dims-filter_kernel_dims)/stride)+1)
        self.input_dims = input_dims
        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        self.OutList = np.zeros((filter_num,self.output_dims,self.output_dims))
        self.temp = np.zeros((self.output_dims,self.output_dims))

    def forward(self,input):
        self.temp = np.zeros((self.output_dims,self.output_dims))

        #wish this was vectorized, but nonetheless, stride multiplication happening her
        for i in range(self.output_dims):
            for j in range(self.output_dims):
                region = input[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims]
                self.temp[i][j] = np.sum(region) / (self.filter_kernal_dims ** 2)
                #pooling layer here is just an average, will change most likely upon further research of well working architectures
        return self.temp
    
    def backwardMatrices(self,input,delta_slice):
        #this is called k times where k = the dimensions of our input layer
        #delta cooresponds to the delL/del p_o/p
        temp = np.zeros((self.input_dims,self.input_dims))

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                max_value = 1e+10  # or some other value based on your expected range
                delta_value = delta_slice[i, j] / pow(self.filter_kernal_dims, 2)
                delta_value = np.clip(delta_value, -max_value, max_value)

                temp[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]+=(delta_value)
    
        return temp

# Load the MNIST dataset from IDX format
def load_mnist():
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print(os.listdir(path))

    print("Path to dataset files:", path)
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    # Load the MNIST data
    train_images = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path) 

    print(f"Training Images Shape: {train_images.shape}")
    print(f"Training Labels Shape: {train_labels.shape}")
    print(f"Training Images Shape: {test_images.shape}")
    print(f"Training Labels Shape: {test_labels.shape}")

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def clip_norm(grad_vector,clipper_val):
    L2_gradV = np.linalg.norm(grad_vector)
    print("norm value: ", L2_gradV)
    if L2_gradV > clipper_val:
        print("had to clip playa, gradients are exploding")
        clipped_grad = (clipper_val/L2_gradV) * grad_vector
    else:
        clipped_grad = grad_vector
    return clipped_grad

def main():
    pass

if __name__=="__main__":
    #TODO write backpropogation logic for Neural Networks{activation functions, weight matrices} && CNN{activation functions}: 1 day
        #then the debug step: max 2 days then its mf done!!!
    # Define the transformation for MNIST data

    train_images, train_labels, test_images, test_labels=load_mnist()
    epochs = 3

    #haha calling main w no code
    main()
    # self,input_dims, filter_kernel_dims, filter_num, stride
    o =outputFilter1(28,8,1,1)
    temp = o.output_dims
    #self,input_dims, filter_kernel_dims, filter_num, stride
    # print("temp is!", temp)
    # print("o1 output dims",o1.output_dims)
    # print("o1 output1 dims", o1.output_dims)
    #pooling layer depth must be equivalent to input depth
    
    #length is defined as a output square stretched times number of channels
    # print("p1.outlist shape:", p1.OutList.shape)
    # print("p1 output dims:", p1.output_dims)
    length = o.output_dims * o.output_dims
    nn_input = np.zeros((0,1))
    currLossMatrix = np.ndarray((o.output_dims,o.output_dims))
    lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))

    n2 = NeuralNetLayer2(length, 10)



    for z in range(epochs):
        correct_predictions = 0
        for i in range(len(train_images)):
                print(i)
                if i%10 == 0 and i >10:
                    print(f"epoch: {z+1},iteration: {i}")
                    print("training accuracy: ", correct_predictions/i)

                n2.y_actual = train_labels[i]
                n2.y_actual_index =  n2.y_actual

                ##loops through number of channels for each layer
                    # print("computing channel {%d}",k)
                o.OutList[0]=o.forward(train_images[i],o.filterList[0],0)
                flat = o.OutList[0].flatten().reshape(-1, 1)
                # print("flat shape: ",flat.shape)
                nn_input = np.concatenate((nn_input, flat), axis=0)
                # print("normal shape", nn_input.shape)
                # print("got past n1")
                n2.forward(nn_input)

                # print("got past n2")
                n2.softmaxxer()
                if np.argmax(n2.outputList) == train_labels[i]:
                    correct_predictions +=1
                    print("correct prediction:", correct_predictions) 

                output_loss=n2.calc_loss()
                lossVector1 = n2.backPropogateCalc()
                # print(lossVector1)
                print("lossVector1")
                lossVector1 = clip_norm(lossVector1, 100)

                # print("SHAPE CHECK OF delLoss/delNNInput",lossVector0.shape)

                #add backpropogation logic for each channel weight!


                print("shape checker",lossVector1.shape)
                for y in range(o.output_dims):
                    for z in range(o.output_dims):
                        currLossMatrix[y][z] = lossVector1[y*o.output_dims + z].item()

                    # print("shape check of delLoss/delNNInput_slice:",currLossMatrix.shape)
                    # print("checking for delta shapes: ",lossesArray.shape,o1.OutList.shape)
                
                    # print("backprop check",o.filterList.shape,data[i].shape,lossesArray2.shape)
                print("currLossMatrix")
                currLossMatrix=clip_norm(currLossMatrix, 100)
                o.filterList[0] -= o.backPropogateWeights(train_images[i],currLossMatrix)

                # print("backpropogated outputfilter")

                print("backpropogated Convolutional weights!")
                n2.backPropogateWeights()

                # print("gradient monitoring",lossesArray,lossesArray1,lossesArray2)
                nn_input = np.zeros((0,1))
                lossesArray = np.zeros((o.filter_num,o.input_dims,o.input_dims))
                lossesArray1 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
                lossesArray2 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
                lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
                lossVector1 = None
                lossVector2 = None
                currLossMatrix = np.ndarray((o.output_dims,o.output_dims))


    correct_test_predictions= 0

    for i in range(len(test_images)):
        print(i)
        if i%10 == 0 and i >10:
            print(f"epoch: {i}")
            print("testing accuracy: ", correct_test_predictions/i)

        n2.y_actual = test_labels[i]
        n2.y_actual_index =  n2.y_actual

        ##loops through number of channels for each layer
            # print("computing channel {%d}",k)
        o.OutList[0]=o.forward(test_images[i],o.filterList[0],0)
        flat = o.OutList[0].flatten().reshape(-1, 1)
        # print("flat shape: ",flat.shape)
        nn_input = np.concatenate((nn_input, flat), axis=0)
        # print("normal shape", nn_input.shape)
        # print("got past n1")
        n2.forward(nn_input)

        # print("got past n2")
        n2.softmaxxer()
        if np.argmax(n2.outputList) == test_labels[i]:
            correct_test_predictions +=1
            print("correct prediction:", correct_test_predictions) 


        output_loss=n2.calc_loss()
        lossVector1 = n2.backPropogateCalc()
        # print(lossVector1)
        print("lossVector1")
        lossVector1 = clip_norm(lossVector1, 100)

        # print("SHAPE CHECK OF delLoss/delNNInput",lossVector0.shape)

        #add backpropogation logic for each channel weight!


        print("shape checker",lossVector1.shape)
        for y in range(o.output_dims):
            for z in range(o.output_dims):
                currLossMatrix[y][z] = lossVector1[y*o.output_dims + z].item()

            # print("shape check of delLoss/delNNInput_slice:",currLossMatrix.shape)
            # print("checking for delta shapes: ",lossesArray.shape,o1.OutList.shape)
        
            # print("backprop check",o.filterList.shape,data[i].shape,lossesArray2.shape)
        print("currLossMatrix")
        currLossMatrix=clip_norm(currLossMatrix, 100)
        o.filterList[0] -= o.backPropogateWeights(train_images[i],currLossMatrix)

        # print("backpropogated outputfilter")

        print("backpropogated Convolutional weights!")
        n2.backPropogateWeights()

        # print("gradient monitoring",lossesArray,lossesArray1,lossesArray2)
        nn_input = np.zeros((0,1))
        lossesArray = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossesArray1 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossesArray2 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossVector1 = None
        lossVector2 = None
        currLossMatrix = np.ndarray((o.output_dims,o.output_dims))
    print("testing accuracy: ", correct_test_predictions/len(test_images))