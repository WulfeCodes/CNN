import numpy as np

#TODO write out the dimensions and do the formulation on white board for locked in understanding
#done with no bias

class NeuralNetLayer1:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layerList = np.ndarray(1,input_dims,64)
        self.outputList = np.ndarray(1,64)
        self.pre_outputList = np.ndarray(1,64)
        self.differentiated_outputList = np.ndarray(1,64)
        self.init_cached_weightMatrices()
        self.input = None
    def init_cached_weightMatrices(self):
        
        for i in range(64):
            temp = np.random.randn(1,*self.input_dims)
            self.layerList[i]=temp
            
    def forward(self,input):
        self.input = input
        for i in range(64):
            self.outputList[i].append(self.layerList[i]@input)

            for j in range(len(self.outputList[i])):
                self.pre_outputList=self.outputList
                self.outputList[i,j]=self.leaky_relu_inline(self.outputList[i,j])

    def leaky_relu_inline(self,x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    def backPropogate(self):
        self.losser_vector=np.ndarray(64,1)

        for i in range(64):
            if self.pre_outputList[i]>0:
                self.losser_vector[i]=self.pre_outputList[i]
            else:
                self.losser_vector[i]=.01

        self.differentiated_outputList=self.pre_outputList * self.outputList.T
        #size 64x1
        return self.differentiated_outputList * self.input.T
        

class NeuralNetLayer2:
    #TODO write in loss and latter layers
    #input is a nx1 vector, there should be 64 weights that transform
        #into a vector of 64x1
    def __init__(self,input_dims,output_dims):

        self.input_dims = input_dims
        self.output_dims = output_dims
        #64 10x1 vectors
        self.layerList = np.ndarray(10,1,64)
        self.outputList = np.ndarray(10,1)
        self.pre_outputList = np.ndarray(10,1)
        self.y_actual = None
        self.init_cached_weightMatrices()
        self.jacobian_sigmoid = np.ndarray(10,10)

    def init_cached_weightMatrices(self):
        
        for i in range(10):
            temp = np.random.randn(1,*self.input_dims)
            self.layerList[i] = temp
    
    def forward(self,input):
        for i in range(64):
            self.outputList[i].append(self.layerList[i]@input)
    
    def softmaxxer(self):

        #nice, finally a vectorized computation!
        self.pre_outputList=self.outputList
        self.outputList = np.exp(self.outputList) / np.sum(np.exp(self.outputList))


    def calc_loss(self,y_actual):
        self.y_actual = y_actual
        #cross entropy loss function: 
        #returning the derived version
        return -(np.log(self.outputList[y_actual-1]))
    
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


    
    def backPropogate(self):
        loss_step1 = -1/np.log(self.outputList[self.y_actual-1])
        self.calc_sigmoid_activation_loss()

        sigmoid_loss_vector = np.ndarray(10,1)
        hidden_weight_loss_vector = np.ndarray(10,1)

        for i in range(10):
            sigmoid_loss_vector[i] = loss_step1 * self.jacobian_sigmoid[i,self.y_actual-1]
            
        #update each weight matrix
        #sigmoid loss vector is of 10x1 which represents the differentiated output by the sigmoid and the loss, leaving the weight operation of the prev layer being the next 
        for i in range(10):
           hidden_weight_loss_vector[i] = sigmoid_loss_vector[i] * self.pre_outputList[i].T
            
        return hidden_weight_loss_vector
    #takes in a 64x1 vector,
    #multiplied by a 1x64 weight, by 10 weights
 



class outputFilter1:
    def __init__(self,input_dims, filter_kernel_dims, filter_num ,stride):
        self.output_dims = (np.floor(input_dims-filter_kernel_dims)/stride)+1

        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        self.filter0 = np.random.randn(*self.output_dims)
        self.filter1 = np.random.randn(*self.output_dims)
        self.filterList = np.ndarray(filter_num,self.output_dims,self.output_dims)
        self.OutList = np.ndarray(filter_num,self.output_dims,self.output_dims)

        self.temp = np.zeros(*self.output_dims)
    def forward(self,input,filter):
        self.temp = np.zeros(*self.output_dims)

        #wish this was vectorized, but nonetheless, stride multiplication happening her
        for i in range(self.output_dims):
            for j in range(self.output_dims):


                self.temp[i][j]=self.leaky_relu_inline(np.sum(input[i:self.filter_kernal_dims][j:self.filter_kernal_dims]*filter))

        return self.temp
    
    def leaky_relu_inline(self,x, alpha=0.01):
        return x if x > 0 else alpha * x
    
    def backwardMatrices(self,input,losses):
        
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #range of losses equates to stretched dimensions of output_dims w each index being one entry
        self.temp = np.zeros(*self.output_dims)

        for i in range(self.output_dims):
            for j in range(self.output_dims):
            
                self.temp=input[j:self.filter_kernal_dims][k:self.filter_kernal_dims]
                losses.append(self.temp)

    
        return losses
    

    

    

class poolingLayer1:
    def __init__(self,input_dims, filter_kernel_dims,filter_num,stride):
        self.output_dims = (np.floor(input_dims-filter_kernel_dims)/stride)+1

        self.filter_num = filter_num
        self.filter_kernal_dims = filter_kernel_dims
        #size 10 filter
        self.OutList = np.ndarray(filter_num,self.output_dims,self.output_dims)


        self.temp = np.zeros(*self.output_dims)
    def forward(self,input):
        self.temp = np.zeros(*self.output_dims)

        #wish this was vectorized, but nonetheless, stride multiplication happening her
        for i in range(self.output_dims):
            for j in range(self.filter_kernal_dims):

                #pooling layer here is just an average, will change most likely upon further research of well working architectures
                self.temp[i][j]=((np.sum(input[i:self.filter_kernal_dims][j:self.filter_kernal_dims]*filter))/self.filter_kernal_dims*self.filter_kernal_dims)
        return self.temp
    
    def backwardMatrices(self,input,losses):
        
        #wish this was vectorized, but nonetheless, stride multiplication happening her
        #range of losses equates to stretched dimensions of output_dims w each index being one entry
        self.temp = np.zeros(*self.output_dims)

        for i in range(self.output_dims):
            for j in range(self.output_dims):
            
                self.temp=input[j:self.filter_kernal_dims][k:self.filter_kernal_dims]
                losses.append(self.temp)

    
        return losses


def main():
    pass

if __name__=="__main__":
    #TODO write backpropogation logic for Neural Networks{activation functions, weight matrices} && CNN{activation functions}: 1 day
        #then the debug step: max 2 days then its mf done cracker!!!
    #imported csv from MNIST
    data = None
    dataset_range = None
    main()
    o =outputFilter1(28,3,2,1)
    temp = o.filter_kernal_dims
    p = poolingLayer1(temp,2,2,1)
    temp = p.filter_kernal_dims
    o1 =outputFilter1(temp,3,2,1)
    temp = o1.filter_kernal_dims
    p1 = poolingLayer1(temp,2,4,1)

    #TODO define length as parameter of final pooling layer!!!
    
    length = p1.output_dims * 2
    nn_input = np.array(length,1)
    lossesArray = np.ndarray(None,None,None)
    lossesArray1 = np.ndarray(None,None,None)
    lossesArray2 = np.ndarray(None,None,None)
    lossesArray3 = np.ndarray(None,None,None)

    n1 = NeuralNetLayer1(length, 64)
    n2 = NeuralNetLayer2(10, 1)

    for i in range(dataset_range):
        #feed forward logic heyr
        
        for k in range(o.filter_num):
            o.OutList[k]=o.forward(data[i],o.filterList[k])

        for k in range(p.filter_num):
            p.OutList[k]=p.forward(o.OutList[k])

        for k in range(o1.filter_num):
            o1.OutList[k]=o.forward(p.OutList[k],o1.filterList[k])

        for k in range(p1.filter_num):
            p1.OutList[k]=p1.forward(o1.OutList[k])
            np.concat(nn_input,np.flatten(p1.OutList[k]))

        n1.forward(nn_input)
        n2.forward(n1.outputList)
        n2.softmaxxer()
        output_loss=n2.calc_loss(data[i][1])

        #TODO actually fix neural network's backward!!
        lossVector1 = n2.backPropogate()
        lossVector0 = n1.backPropogate()
        #TODO DISCONNECT HERE

        #BACKPROPOGATION DOWN HERE!!
        #NOT DONE RIGHT AT FUCKING ALL!!!
        for k in range(p1.filter_num):
            lossesArray.append(p1.backwardMatrices(o1.OutList[k],lossVector0))

        for k in range(o1.filter_num):
            lossesArray1 = o1.backwardMatrices(p.OutList[k],lossesArray)

        for k in range(p.filter_num):
            lossesArray2 = p.backwardMatrices(o.OutList[k],lossesArray1)

        for k in range(o.filter_num):
           lossesArray3 = o.backwardMatrices(data[1],lossesArray2)
        
        #feed forward into neural network

    #z_i_j = w_j * input, differentiated wrt to w_j you get the input multiplied by some alpha
    #leaky relu differentiated is 1 if x>1 and a if less than 0?
    #figure out how to backpropogate wrt to each filter


        

        

    

    #store list of matrices length of filter_num w dimensions of output_dims