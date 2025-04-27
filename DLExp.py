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

        ##haha 64 is hard coded like fuck here
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layerList = np.random.rand(64,input_dims) * np.sqrt(2/input_dims)
        self.outputList = np.ndarray((64,1))
        self.pre_outputList = np.ndarray((64,1))
        self.differentiated_outputList = np.ndarray((64,1))
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
        self.layerList -= (0.002 * delta2 @ self.input.T)
        print("updated the weights!")
        

class NeuralNetLayer2:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        #64 10x1 vectors.T, changed from 10,1,64 deleted batch dim
        self.layerList = np.random.rand(10,64) * np.sqrt(2/64)
        self.outputList = np.ndarray((10,1))
        self.pre_outputList = np.ndarray((10,1))
        #temp y_actual, should be passed in a parameter. thatll be known once I analyze the mnist dataset
        self.y_actual = 0
        self.sum = None
        self.y_actual_index = self.y_actual -1
        # self.init_cached_weightMatrices()
        self.jacobian_sigmoid = np.ndarray((10,10))
        self.differentiated_outputVector = np.zeros((10,1))
        self.delta2 = np.zeros((10,1))
        self.delta1 = np.zeros((10,1))
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
        print("here are the one hot encoded preds and its adjusted val", val, safe_val)
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
        self.layerList -= (0.002 * self.delta2 @ self.input.T)
        print("layer weights 2 updated!")
 
class outputFilter1:
    def __init__(self,input_dims, filter_kernel_dims, filter_num , stride):
        self.output_dims = int((np.floor(input_dims-filter_kernel_dims)/stride)+1)
        self.input_dims = input_dims
        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        ##TODO change randomization to be of xavier/fan in variant 
        
        self.filterList = np.zeros((filter_num,self.filter_kernal_dims,self.filter_kernal_dims))

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
        
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #range of losses equates to stretched dimensions of output_dims w each index being one entry
        temp = np.zeros((self.input_dims,self.input_dims))
        #TODO sumn about this shape calculation aint right playa, since theres upsampling shouldnt there be a padding of some sort?

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                # print("should be of some shape", input.shape)
                # print("should be of kernal shape:", (temp[i:self.filter_kernal_dims,j:self.filter_kernal_dims]).shape)
                # print("should be a scalar",losses[i,j].shape)
                #differentiated to create delta meaning the left over would be the weight kernal!
                temp[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims] += kernel * (losses[i,j]* self.leaky_relu_derivative(index[i,j]))
                #matrix multiplied with a scalar
        # print("we finished computing temp!", temp)
        return temp
    
    def backPropogateWeights(self,input,lossDelta):
        temp = np.zeros((self.filter_kernal_dims,self.filter_kernal_dims))
        # print("tempest shape: ",temp.shape)
        for i in range(self.output_dims):
            for j in range(self.output_dims):
                # print("loss delta shape", lossDelta[i,j].shape)
                #CHANGED THIS TO ADD THE ASSOCIATED DERIVATIVE WRT TO THE INPUT THAT WOULDVE OCCURED INSTEAD OF THE losses index like b4
                region = input[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims]
                region_sum = np.sum(region)
                # region_sum = np.clip(region_sum, -1e2, 1e2)  # clamp huge values

                derivative = self.leaky_relu_derivative(region_sum)
                delta_val = lossDelta[i, j] * derivative
                scaled_region = region * delta_val

                # prevent overflow
                # scaled_region = np.clip(scaled_region, -1e2, 1e2)

                temp += 0.002 * scaled_region
        # print("finna return temp now", temp.shape)         
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
        
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #range of losses equates to stretched dimensions of output_dims w each index being a scalar
        temp = np.zeros((self.input_dims,self.input_dims))
        # print("delta one to many input shape check:", delta_slice.shape)
        # print("delta one to many input type check:", delta_slice.dtype)

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                #one to many gradient wrt to some delL/delY_i,j
                #changed from the input being accounted for in the summation, if differentiating wrt to input i will just be delta[i,j]/k^2 
                # print("everything must go!")
                # print((temp[i:self.filter_kernal_dims][j:self.filter_kernal_dims]).shape)
                # print(delta_slice[i,j].shape)
                # print(self.output_dims)
                # print(delta_slice.shape)
                # print(i+j)
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
    epochs = 60000

    #haha calling main w no code
    main()
    # self,input_dims, filter_kernel_dims, filter_num, stride
    o =outputFilter1(28,3,4,1)
    temp = o.output_dims
    #self,input_dims, filter_kernel_dims, filter_num, stride
    p = poolingLayer1(temp,2,4,1)
    temp = p.output_dims
    # print("temp is!", temp)
    o1 =outputFilter1(temp,3,2,1)
    # print("o1 output dims",o1.output_dims)
    temp = o1.output_dims
    p1 = poolingLayer1(temp,2,2,1)
    # print("o1 output1 dims", o1.output_dims)
    #pooling layer depth must be equivalent to input depth
    
    #length is defined as a output square stretched times number of channels
    # print("p1.outlist shape:", p1.OutList.shape)
    # print("p1 output dims:", p1.output_dims)
    length = p1.output_dims * p1.filter_num * p1.output_dims
    nn_input = np.zeros((0,1))
    lossesArray = np.zeros((p1.filter_num,p1.input_dims,p1.input_dims))
    lossesArray1 = np.zeros((p.filter_num,o1.input_dims,o1.input_dims))
    lossesArray2 = np.zeros((p.filter_num,p.input_dims,p.input_dims))
    lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))

    n1 = NeuralNetLayer1(length, 64)
    n2 = NeuralNetLayer2(10, 1)

    correct_predictions = 0

    for i in range(epochs):
        if i>10 and i%10 == 0:
            print(f"epoch: {i}")
            print("training accuracy: ", correct_predictions/i)

        n2.y_actual = train_labels[i]
        n2.y_actual_index =  n2.y_actual
        #this needs to be set according to the dataset!
        # print("got to here")
        #feed forward logic heyr
        
        ##loops through number of channels for each layer
        for k in range(o.filter_num):
            # print("computing channel {%d}",k)
            o.OutList[k]=o.forward(train_images[i],o.filterList[k],k)

        for k in range(p.filter_num):
            # print("WE POOLING")
            p.OutList[k]=p.forward(o.OutList[k])

        # print("o shape:", p.OutList.shape)

        # print("debugging filter numbers: p1 then o1", p1.filter_num,o1.filter_num)

        #this convolution operation is done by the summation of the convolution operations of each previous filter by one convo Kernal
        for k in range(o1.filter_num):
            tempMatrix = np.zeros((o1.output_dims,o1.output_dims))
            for z in range(p.filter_num):
                # print("convo layer numba 2")
                # # print(o1.output_dims)
                # print("respected shapes: ", p.OutList[z].shape, o1.filterList[k].shape,tempMatrix.shape)
                tempMatrix+=o1.forward(p.OutList[z],o1.filterList[k],k)
                # print("pooling 1 number",p1.filter_num)
            o1.OutList[k]=tempMatrix
        # print("o1 outlist shape",o1.OutList.shape)

        # print("nn input shape before the accident:", nn_input.shape)
        for k in range(p1.filter_num):
            # print("we pooling numba 2!",k)
            p1.OutList[k]=p1.forward(o1.OutList[k])
            flat = p1.OutList[k].flatten().reshape(-1, 1)
            # print("flat shape: ",flat.shape)
            nn_input = np.concatenate((nn_input, flat), axis=0)

        # print("normal shape", nn_input.shape)
        n1.forward(nn_input)
        # print("got past n1")
        n2.forward(n1.outputList)

        # print("got past n2")
        n2.softmaxxer()

        if np.argmax(n2.outputList) == train_labels[i]:
            correct_predictions +=1
            print("correct prediction:", correct_predictions) 


        output_loss=n2.calc_loss()

        lossVector1 = n2.backPropogateCalc()

        print("lossVector 1: ")
        # print(lossVector1)
        lossVector1 = clip_norm(lossVector1, 100)

        lossVector0 = n1.backPropogateCalc(lossVector1)

        print("lossVector 0: ")

        lossVector0 = clip_norm(lossVector0, 100)
        # print("SHAPE CHECK OF delLoss/delNNInput",lossVector0.shape)

        #add backpropogation logic for each channel weight!
        currLossMatrix = np.ndarray((p1.filter_num,p1.output_dims,p1.output_dims))

        #this should be good for backpropogating along each input by their contributing quantity by avg: 1/k^2 * DelL/Del(corresponding output slice)
        #haha as of rn, currLossMatrix is implemented horribly wrong, how do I change this creatively?
            #right now its only slicing through 2*k slices, where it should be slicing k^2 input slices! 
            #[lossVector0[k*p1.output_dims:(k+1)*pow(p1.output_dims,2)]]
        for k in range(p1.filter_num):
            #currLossMatrix initializer
            for y in range(p1.output_dims):
                for z in range(p1.output_dims):
                    index = (k * pow(p1.output_dims,2)) + ((y * p1.output_dims) + z)
                    currLossMatrix[k][y][z] = lossVector0[index].item()

            # print("shape check of delLoss/delNNInput_slice:",currLossMatrix.shape)
            lossesArray[k]=p1.backwardMatrices(o1.OutList[k],currLossMatrix[k])
            # print("checking for delta shapes: ",lossesArray.shape,o1.OutList.shape)
        
#OR>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #but would the losses array be respecitvely associated with this result? I say yes because it is linearly associated with each filter keeping the mapping respective
        #TODO I think this implementation is wrong, there should be  
        for i in range(p.filter_num):
            for j in range(o1.filter_num):
                #should pass in o1.outList here
                lossesArray1[i] +=o1.backwardMatrices(o1.filterList[j],lossesArray[j],o1.preOutList[j])
                

        print("lossArray 1: ")
        lossesArray1 = clip_norm(lossesArray1,100)
            #input dims is of p.filter_num, p.output_dims or o1.input_dims x p.output_dims or o1.input_dims
            #each output list should correspond to a one to many, meaning many to one output matrices will update the outputList

        # print("checking for delta shapes1: ",lossesArray1.shape,p.OutList.shape)
        for k in range(p.filter_num):
            lossesArray2[k]=p.backwardMatrices(o.OutList[k],lossesArray1[k])

        print("lossArray2: ")
        lossesArray2 = clip_norm(lossesArray2,100)

        for k in range(o.filter_num):
            # print("backprop check",o.filterList.shape,data[i].shape,lossesArray2.shape)
            o.filterList[k] -= o.backPropogateWeights(train_images[i],lossesArray2[k])
        # print("backpropogated outputfilter")


        temp = np.zeros_like(o1.filterList[0])
        print(o1.filterList.shape,o1.filter_num)
        # print("checking losses shape, dont mind me",lossesArray.shape)
        for j in range(o1.filter_num):
            temp = np.zeros_like(o1.filterList[0])
            for k in range(2):
                temp+=o1.backPropogateWeights(p.OutList[k],lossesArray[j])
            o1.filterList[j] -= temp

        print("backpropogated Convolutional weights!")
        n1.backPropogateWeights(lossVector1)
        n2.backPropogateWeights()


        # print("gradient monitoring",lossesArray,lossesArray1,lossesArray2)
        nn_input = np.zeros((0,1))
        lossesArray = np.zeros((p1.filter_num,p1.input_dims,p1.input_dims))
        lossesArray1 = np.zeros((p.filter_num,o1.input_dims,o1.input_dims))
        lossesArray2 = np.zeros((p.filter_num,p.input_dims,p.input_dims))
        lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossVector1 = None
        lossVector2 = None


    correct_test_predictions= 0

    for i in range(len(test_images)):
        if i%10 == 0 and i >10:
            print(f"epoch: {i}")
            print("testing accuracy: ", correct_test_predictions/i)

        n2.y_actual = test_labels[i]
        n2.y_actual_index =  n2.y_actual

        ##loops through number of channels for each layer
        for k in range(o.filter_num):
            # print("computing channel {%d}",k)
            o.OutList[k]=o.forward(train_images[i],o.filterList[k],k)

        for k in range(p.filter_num):
            # print("WE POOLING")
            p.OutList[k]=p.forward(o.OutList[k])

        # print("o shape:", p.OutList.shape)

        # print("debugging filter numbers: p1 then o1", p1.filter_num,o1.filter_num)

        #this convolution operation is done by the summation of the convolution operations of each previous filter by one convo Kernal
        for k in range(o1.filter_num):
            tempMatrix = np.zeros((o1.output_dims,o1.output_dims))
            for z in range(p.filter_num):
                # print("convo layer numba 2")
                # # print(o1.output_dims)
                # print("respected shapes: ", p.OutList[z].shape, o1.filterList[k].shape,tempMatrix.shape)
                tempMatrix+=o1.forward(p.OutList[z],o1.filterList[k],k)
                # print("pooling 1 number",p1.filter_num)
            o1.OutList[k]=tempMatrix
        # print("o1 outlist shape",o1.OutList.shape)

        # print("nn input shape before the accident:", nn_input.shape)
        for k in range(p1.filter_num):
            # print("we pooling numba 2!",k)
            p1.OutList[k]=p1.forward(o1.OutList[k])
            flat = p1.OutList[k].flatten().reshape(-1, 1)
            # print("flat shape: ",flat.shape)
            nn_input = np.concatenate((nn_input, flat), axis=0)

        # print("normal shape", nn_input.shape)
        n1.forward(nn_input)
        # print("got past n1")
        n2.forward(n1.outputList)

        # print("got past n2")
        n2.softmaxxer()

        if np.argmax(n2.outputList) == train_labels[i]:
            correct_predictions +=1
            print("correct prediction:", correct_predictions) 


        output_loss=n2.calc_loss()

        lossVector1 = n2.backPropogateCalc()

        print("lossVector 1: ")
        # print(lossVector1)
        lossVector1 = clip_norm(lossVector1, 100)

        lossVector0 = n1.backPropogateCalc(lossVector1)

        print("lossVector 0: ")

        lossVector0 = clip_norm(lossVector0, 100)
        # print("SHAPE CHECK OF delLoss/delNNInput",lossVector0.shape)

        #add backpropogation logic for each channel weight!
        currLossMatrix = np.ndarray((p1.filter_num,p1.output_dims,p1.output_dims))

        #this should be good for backpropogating along each input by their contributing quantity by avg: 1/k^2 * DelL/Del(corresponding output slice)
        #haha as of rn, currLossMatrix is implemented horribly wrong, how do I change this creatively?
            #right now its only slicing through 2*k slices, where it should be slicing k^2 input slices! 
            #[lossVector0[k*p1.output_dims:(k+1)*pow(p1.output_dims,2)]]
        for k in range(p1.filter_num):
            #currLossMatrix initializer
            for y in range(p1.output_dims):
                for z in range(p1.output_dims):
                    index = (k * pow(p1.output_dims,2)) + ((y * p1.output_dims) + z)
                    currLossMatrix[k][y][z] = lossVector0[index].item()

            # print("shape check of delLoss/delNNInput_slice:",currLossMatrix.shape)
            lossesArray[k]=p1.backwardMatrices(o1.OutList[k],currLossMatrix[k])
            # print("checking for delta shapes: ",lossesArray.shape,o1.OutList.shape)
        
#OR>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #but would the losses array be respecitvely associated with this result? I say yes because it is linearly associated with each filter keeping the mapping respective
        #TODO I think this implementation is wrong, there should be  
        for i in range(p.filter_num):
            for j in range(o1.filter_num):
                #should pass in o1.outList here
                lossesArray1[i] +=o1.backwardMatrices(o1.filterList[j],lossesArray[j],o1.preOutList[j])
                

        print("lossArray 1: ")
        lossesArray1 = clip_norm(lossesArray1,100)
            #input dims is of p.filter_num, p.output_dims or o1.input_dims x p.output_dims or o1.input_dims
            #each output list should correspond to a one to many, meaning many to one output matrices will update the outputList

        # print("checking for delta shapes1: ",lossesArray1.shape,p.OutList.shape)
        for k in range(p.filter_num):
            lossesArray2[k]=p.backwardMatrices(o.OutList[k],lossesArray1[k])

        print("lossArray2: ")
        lossesArray2 = clip_norm(lossesArray2,100)

        for k in range(o.filter_num):
            # print("backprop check",o.filterList.shape,data[i].shape,lossesArray2.shape)
            o.filterList[k] -= o.backPropogateWeights(train_images[i],lossesArray2[k])
        # print("backpropogated outputfilter")

        temp = np.zeros_like(o1.filterList[0])
        print(o1.filterList.shape,o1.filter_num)
        # print("checking losses shape, dont mind me",lossesArray.shape)
        for j in range(o1.filter_num):
            temp = np.zeros_like(o1.filterList[0])
            for k in range(2):
                temp+=o1.backPropogateWeights(p.OutList[k],lossesArray[j])
            o1.filterList[j] -= temp

        print("backpropogated Convolutional weights!")
        n1.backPropogateWeights(lossVector1)
        n2.backPropogateWeights()

        # print("gradient monitoring",lossesArray,lossesArray1,lossesArray2)
        nn_input = np.zeros((0,1))
        lossesArray = np.zeros((p1.filter_num,p1.input_dims,p1.input_dims))
        lossesArray1 = np.zeros((p.filter_num,o1.input_dims,o1.input_dims))
        lossesArray2 = np.zeros((p.filter_num,p.input_dims,p.input_dims))
        lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))
        lossVector1 = None
        lossVector2 = None