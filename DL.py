import numpy as np

#TODO organize channel generalization for convolutional layers
    #currently convolution layer 2 does not have appropriate channel dims, should be 4 instead is 2
        #it should have 2 initialized channels that creates a 2^filter_num of channels in its output list
#done with no bias

class NeuralNetLayer1:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        ##haha 64 is hard coded like fuck here
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layerList = np.random.rand(64,input_dims)
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
        print("hello here are the shapes")
        print(self.layerList.shape)
        print(self.outputList.shape)
        print(input.shape)
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
                self.losser_vector[i]=self.pre_outputList[i]
            else:
                self.losser_vector[i]=.01

        #returning delta1 here, delta2 here,

        # self.differentiated_outputList=self.pre_outputList * self.outputList.T
        #size 64x1
        print("self layer list dimensions:, should be of 64 x input_dims",self.layerList.shape)
        print("delta dimensions: should be of 64x1",delta2.shape)
        print("losser vector should be of 64x1 shape", self.losser_vector.shape)

        return self.layerList.T @ (delta2 * self.losser_vector)
        #self.LayerList is 64 x input_dims times delta 2 which is 64x1 equaling our input dim of 1936x1

    def backPropogateWeights(self,delta2):
        #partial of H_w{i}=input_i@deltai+1, for here it should equal 64 x 1936 dims
        #delta2 is 64x1 self.input is 1936x1
        self.layerList -= (0.1 * delta2 @ self.input.T)
        print("updated the weights!")
        

class NeuralNetLayer2:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        #64 10x1 vectors.T, changed from 10,1,64 deleted batch dim
        self.layerList = np.random.rand(10,64)
        self.outputList = np.ndarray((10,1))
        self.pre_outputList = np.ndarray((10,1))
        #temp y_actual, should be passed in a parameter. thatll be known once I analyze the mnist dataset
        self.y_actual = int(9)
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
        self.outputList = np.exp(self.outputList) / np.sum(np.exp(self.outputList))


    def calc_loss(self,y_actual):
        self.y_actual = y_actual
        #cross entropy loss function: 
        #returning the derived version
        self.differentiated_outputVector[self.y_actual_index] = -(np.log(self.outputList[self.y_actual_index]))
    
    def calc_sigmoid_activation_loss(self):
        #derivation of jacobian is given by a(1-a) for diagonal elements
        #-a_ia_j for off diagonal elements
        #try to switch these to vectorized computations
        for i in range(10):
            for j in range(10):
                if i==j:
                    self.jacobian_sigmoid[i,j]=self.outputList[i]*(1-self.outputList[i])

                else:
                    self.jacobian_sigmoid[i,j]=-(self.outputList[i]*self.outputList[j])


    
    def backPropogateCalc(self):
        #i dont think there is a negative here, if its just off of cross entropy/expected likelihood
        loss_step1 = -1/np.log(self.outputList[self.y_actual_index])
        self.calc_sigmoid_activation_loss()

        sigmoid_loss_vector = np.ndarray((10,1))
        hidden_weight_loss_vector = np.ndarray((10,1))

        for i in range(10):
            sigmoid_loss_vector[i] = self.jacobian_sigmoid[i,self.y_actual_index]
        
        #backpropogate to create delta2: dL/dX2 = sigmoid_loss_vector ⊙ differentiated_lossVector
        #delta 2 is 10x1, delta 1 is 64x1, self.LayerList is 10x64
        self.delta2 = sigmoid_loss_vector * self.differentiated_outputVector
        self.delta1 = self.layerList.T @ self.delta2 

        #update each weight matrix
        #sigmoid loss vector is of 10x1 which represents the differentiated output by the sigmoid and the loss, leaving the weight operation of the prev layer being the next 
        # for i in range(10):
        #    hidden_weight_loss_vector[i] = sigmoid_loss_vector[i] * self.pre_outputList[i].T
        #delta 2 will be used for latter loss computations, each delta is used for each layer
        return self.delta1
    
    def backPropogateWeights(self):

    #takes in a 64x1 vector,
    #multiplied by a 10x64 weights, delta2 is 10x1
        self.layerList -= (0.1 * self.delta2 @ self.input.T)
        print("layer weights 2 updated!")
 
class outputFilter1:
    def __init__(self,input_dims, filter_kernel_dims, filter_num ,prev_layer_filter_num, stride):
        self.output_dims = int((np.floor(input_dims-filter_kernel_dims)/stride)+1)
        self.input_dims = input_dims
        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        ##TODO change randomization to be of xavier/fan in variant 

        self.filter0 = np.random.randn(self.filter_kernal_dims,self.filter_kernal_dims)
        self.filter1 = np.random.randn(self.filter_kernal_dims,self.filter_kernal_dims)
        self.filterList = np.zeros((filter_num,self.filter_kernal_dims,self.filter_kernal_dims))
        self.OutList = np.zeros((filter_num*prev_layer_filter_num,self.output_dims,self.output_dims))
        #filterList and output.
        self.temp = np.zeros((self.output_dims,self.output_dims))
    def forward(self,input,filter): 
        self.temp = np.zeros((self.output_dims,self.output_dims))

        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #if stride were to take effect, just change the iterator of i and j for loops
        #TODO decouple leaky relu, would be done by done at backpropogation
        print("output dims equals: ",self.output_dims)

        for i in range(self.output_dims):
            # print("i is ", i)
            for j in range(self.output_dims):
                # print("j is: ",j)
                # print("shape of actual input",input.shape)
                # print("shape of input",input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims].shape)
                # print("shape of filter")
                
                #hadamard product or reg matrix multiplication?
                self.temp[i][j]=self.leaky_relu_inline(np.sum(input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]*filter))

        return self.temp
    
    def leaky_relu_inline(self,x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    def leaky_relu_derivative(self,x,alpha=0.1):
        return 1 if x > 0 else alpha
    
    def backwardMatrices(self,input,losses):
        
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
                temp[i:i+self.filter_kernal_dims, j:j+self.filter_kernal_dims] +=input * (losses[i,j]* self.leaky_relu_derivative(losses[i,j]))
                #matrix multiplied with a scalar
        print("we finished computing temp!", temp)
        return temp
    
    def backPropogateWeights(self,input,lossDelta):
        temp = np.zeros((self.filter_kernal_dims,self.filter_kernal_dims))
        print("tempest shape: ",temp.shape)
        for i in range(self.output_dims):
            for j in range(self.output_dims):
                # print("loss delta shape", lossDelta[i,j].shape)
                temp += 0.1 * (input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims] * (lossDelta[i,j]* self.leaky_relu_derivative(lossDelta[i,j])))
        print("finna return temp now", temp.shape)         
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

                #pooling layer here is just an average, will change most likely upon further research of well working architectures
                self.temp[i][j]=((np.sum(input[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]))/self.filter_kernal_dims*self.filter_kernal_dims)
        return self.temp
    
    def backwardMatrices(self,input,delta_slice):
        
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #range of losses equates to stretched dimensions of output_dims w each index being a scalar
        temp = np.ones((self.input_dims,self.input_dims))
        # print("delta one to many input shape check:", delta_slice.shape)
        print("delta one to many input type check:", delta_slice.dtype)

        for i in range(self.output_dims):
            for j in range(self.output_dims):
                #one to many gradient wrt to some delL/delY_i,j
                #changed from the input being accounted for in the summation, if differentiating wrt to input i will just be delta[i,j]/k^2 
                # print("everything must go!")
                # print((temp[i:self.filter_kernal_dims][j:self.filter_kernal_dims]).shape)
                # print(delta_slice[i,j].shape)
                print(self.output_dims)
                print(delta_slice.shape)
                print(i+j)
                temp[i:i+self.filter_kernal_dims,j:j+self.filter_kernal_dims]*=(delta_slice[i,j]/pow(p1.filter_kernal_dims,2))
    
        return temp


def main():
    pass

if __name__=="__main__":
    #TODO write backpropogation logic for Neural Networks{activation functions, weight matrices} && CNN{activation functions}: 1 day
        #then the debug step: max 2 days then its mf done!!!

    data = np.random.rand(3,28,28)
    #imported csv from MNIST
    dataset_range = 3
    main()
    # self,input_dims, filter_kernel_dims, filter_num ,prev_layer_filter_num, stride
    o =outputFilter1(28,3,2,1,1)
    temp = o.output_dims
    #self,input_dims, filter_kernel_dims, filter_num ,prev_layer_filter_num, stride
    p = poolingLayer1(temp,2,2,1)
    temp = p.output_dims
    print("temp is!", temp)
    o1 =outputFilter1(temp,3,2,p.filter_num,1)
    print("o1 output dims",o1.output_dims)
    temp = o1.output_dims
    p1 = poolingLayer1(temp,2,4,1)
    print("o1 output1 dims", o1.output_dims)
    #TODO define length as parameter of final pooling layer!!!
    
    #length is defined as a output square stretched times number of channels
    print("p1.outlist shape:", p1.OutList.shape)
    print("p1 output dims:", p1.output_dims)
    length = p1.output_dims * 4 * p1.output_dims
    nn_input = np.zeros((0,1))
    lossesArray = np.zeros((p1.filter_num,p1.input_dims,p1.input_dims))
    lossesArray1 = np.zeros((p.filter_num,o1.input_dims,o1.input_dims))
    lossesArray2 = np.zeros((p.filter_num,p.input_dims,p.input_dims))
    lossesArray3 = np.zeros((o.filter_num,o.input_dims,o.input_dims))

    n1 = NeuralNetLayer1(length, 64)
    n2 = NeuralNetLayer2(10, 1)

    for i in range(dataset_range):
        print("got to here")
        #feed forward logic heyr
        
        ##loops through number of channels for each layer
        for k in range(o.filter_num):
            print("computing channel {%d}",k)
            o.OutList[k]=o.forward(data[i],o.filterList[k])

        for k in range(p.filter_num):
            print("WE POOLING")
            p.OutList[k]=p.forward(o.OutList[k])

        print("o shape:", p.OutList.shape)

        #TODO these 2s are hard coded rn, but they need to have a generalized version for ez re iteration for number of filters types
        print("debugging filter numbers: p1 then o1", p1.filter_num,o1.filter_num)

        for k in range(p.filter_num):
            for z in range(o1.filter_num):
                print("convo layer numba 2")
                # print(o1.output_dims)
                o1.OutList[k]=o1.forward(p.OutList[k],o1.filterList[z])
                print("pooling 1 number",p1.filter_num)
        print("o1 outlist shape",o1.OutList.shape)

        print("nn input shape before the accident:", nn_input.shape)
        for k in range(p1.filter_num):
            print("we pooling numba 2!",k)
            p1.OutList[k]=p1.forward(o1.OutList[k])
            flat = p1.OutList[k].flatten().reshape(-1, 1)
            print("flat shape: ",flat.shape)
            nn_input = np.concatenate((nn_input, flat), axis=0)

        print("normal shape", nn_input.shape)
        n1.forward(nn_input)
        print("got past n1")
        n2.forward(n1.outputList)
        print("got past n2")
        n2.softmaxxer()
        output_loss=n2.calc_loss(data[i][1])

        lossVector1 = n2.backPropogateCalc()
        lossVector0 = n1.backPropogateCalc(lossVector1)
        print("SHAPE CHECK OF delLoss/delNNInput",lossVector0.shape)

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
                    currLossMatrix[k][y][z] = lossVector0[index]

            print("shape check of delLoss/delNNInput_slice:",currLossMatrix.shape)
            lossesArray[k]=p1.backwardMatrices(o1.OutList[k],currLossMatrix[k])
            print("checking for delta shapes: ",lossesArray.shape,o1.OutList.shape)
            #lossesArray doesnt backpropogate over the pooling weights, it connects each input to the chain ruled loss function quantity

        for j in range(p.filter_num):
            for k in range(o1.filter_num):
            #TODO does each k coorespond with one another?
                print("New shaper checker", o1.filterList.shape)
                print("new loss checker:",lossesArray.shape)
                if j==0:
                    lossesArray1[j] += o1.backwardMatrices(o1.filterList[j],lossesArray[k])
                else:
                    lossesArray1[j] += o1.backwardMatrices(o1.filterList[j],lossesArray[k+o1.filter_num])
            #input dims is of p.filter_num, p.output_dims or o1.input_dims x p.output_dims or o1.input_dims
            #each output list should correspond to a one to many, meaning many to one output matrices will update the outputList

        print("checking for delta shapes1: ",lossesArray1.shape,p.OutList.shape)
        for k in range(p.filter_num):
            
            lossesArray2[k]=p.backwardMatrices(o.OutList[k],lossesArray1[k])
        print("checking for delta shapes2: ",lossesArray2.shape,o.OutList.shape)
        print("Checking for delta shapes3: ", lossesArray3.shape)
        # for k in range(o.filter_num):
        #     for j in range(o.filter_num):
        #     #Not needed for backprop, if there was a prologue itd be required
        #         lossesArray3[k] += o.backwardMatrices(o.filterList[0],lossesArray[k])

        #backpropogateWeights will only be happening for the convolutional layers, each method will take some input matrix, slide over w the respected delta
        # and update each weight matrix by the summatation of the

        #TODO rough draft code is written out for backpropogation logic, needs some refining ab 1-2 more seshes, filter nums need to be sorted out!
        for k in range(o.filter_num):
            print("backprop check",o.filterList.shape,data[i].shape,lossesArray2.shape)
            o.filterList[k] -= o.backPropogateWeights(data[i],lossesArray2[k])
        print("backpropogated outputfilter")

        print(o1.filterList.shape,o1.filter_num)
        print("checking losses shape, dont mind me",lossesArray.shape)
        for j in range(o1.filter_num):
            for k in range(2):
                if j==0:
                    o1.filterList[j] -= o1.backPropogateWeights(p.OutList[k],lossesArray[k])
                else:
                    o1.filterList[j] -= o1.backPropogateWeights(p.OutList[k],lossesArray[k+2])

        print("backpropogated Convolutional weights!")
        n1.backPropogateWeights(lossVector1)
        n2.backPropogateWeights()

        #this sequence of for loops up above just connects the loss gradient wrt to each input matrix
        #down below is where the backward matrix operation happens!


    #z_i_j = w_j * input, differentiated wrt to w_j you get the input multiplied by some alpha
    #leaky relu differentiated is 1 if x>1 and a if less than 0?
    #figure out how to backpropogate wrt to each filter
        
    #store list of matrices length of filter_num w dimensions of output_dims