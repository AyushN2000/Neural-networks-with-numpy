import numpy as np
from sklearn.preprocessing import StandardScaler

t = 10000000  #Number of training examples
#np.seterr(all='ignore')


def reLU(Z):
    return np.maximum(0,Z)


def subtract_mean(a):
    return np.maximum(a,a-np.mean(a,axis=1,keepdims=True))

def softmax(a):  #Activation function which decides probability of output
    a = scaler.fit_transform(subtract_mean(a))
    return np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)


def accuracy(out,y,m):
    return np.sum(np.sum(out*y))/(5*m)
        

def cost_function(out, Y,m):
    return np.squeeze(-np.sum(np.multiply(Y, np.log(out)) +  np.multiply(1-Y, np.log(1-out)))/m)


def generate_data(m):
    #Genaration of training data
    x = np.random.randint(100, high=1500, size=(m, 3))
    x_row_max = np.max(x, axis=1, keepdims=True)
    below_threshold = np.where(x_row_max <= 180) #A tuple whose first value contains indices where condition is true
    y = np.zeros(shape=(m, 3))
    y = np.equal(x, x_row_max) #Finding out which way to turn based on max distance in front of the robot
    y = y + 0
    y[below_threshold[0]] = [0, 0, 0]
    y = np.concatenate((y, np.zeros(shape=(m, 1))), axis=1)
    y[below_threshold[0], 3] = 1 
    y = np.concatenate((y, np.ones(shape=(m, 1))), axis=1)
    conditions = [(x_row_max >= 1000), (x_row_max >= 500) & (x_row_max < 1000),
              (x_row_max >= 450) & (x_row_max < 500),
              (x_row_max >= 400) & (x_row_max < 450),
              (x_row_max >= 300) & (x_row_max < 400), (x_row_max < 300)]
    choices = [1, 0.85, 0.8, 0.7, 0.6, 0.5]
    y[:, 4] = np.select(conditions, choices).T
    data = {
        "x":x,
        "y":y
    }
    print("Returning data")
    return data


def random_weights_initializer(n_layers,arr_nodes):
    print("Random initializing the weights and biases")
    # Neural Network initialization
    print("n layers : {} and architechture : {} ".format(n_layers,arr_nodes))
    print("initializing W_0 as 3 x {} ",format(arr_nodes[0]))
    parameters = {
        "W_0":np.random.randn(3,arr_nodes[0])
    }
    for i in range(n_layers-1):
        parameters["W_"+str(i+1)] = np.random.randn(arr_nodes[i],arr_nodes[i+1])
        print("initializing W_"+str(i+1)+" as {} x {} ".format(arr_nodes[i],arr_nodes[i+1]))
    print("initializing W_"+str(n_layers)+" as {} x 5",format(arr_nodes[n_layers-1]))
    parameters["W_"+str(n_layers)] = np.random.randn(arr_nodes[n_layers-1],5)
    print("Initializing bias_0 with dimensions : 1 x {} ",format(3))
    parameters["B_0"] = np.ones(shape=(1,3))
    for i in range(n_layers):
        print("Initializing bias_"+str(i+1)+" with dimensions : 1 x {} ",format(arr_nodes[i]))
        parameters["B_"+str(i+1)] = np.ones(shape=(1,arr_nodes[i]))
    print("Initializing bias_"+str(n_layers+1)+" with dimensions : 1 x {} ",format(5))
    parameters["B_"+str(n_layers+1)] = np.ones(shape=(1,5))
    
    
    return parameters


def forward_propagation(d,w_and_b,n_layers):
    #print("Starting feedforward,by intializing weights")
    
    cache = {
        "A_1": scaler.fit_transform(subtract_mean(reLU(np.matmul(d["x"],w_and_b["W_0"])+w_and_b["B_1"])))
    }
    #print("Computing activation A_1 with dimensions :"+str(cache["A_1"].shape))
    for i in range(1,n_layers):
        cache["A_"+str(i+1)] = scaler.fit_transform(subtract_mean(reLU(np.matmul(cache["A_"+str(i)],w_and_b["W_"+str(i)])+w_and_b["B_"+str(i+1)])))
        #print("Computing activation A_"+str(i+1)+" with dimensions :"+str(cache["A_"+str(i+1)].shape))
    cache["out"] = softmax(np.dot(cache["A_"+str(n_layers)],w_and_b["W_"+str(n_layers)]))
    
    return cache


def back_propagation(data,act_l,parameters,n_layers,m):
    grad_Z = {
        "1":act_l["out"] - data["y"]
    }
    #print("Computing gradients grad_Z"+str(1)+" with dimensions :"+str(grad_Z[str(1)].shape))

    for i in range(n_layers):
        grad_Z[str(i+2)] = np.dot(grad_Z[str(i+1)],parameters["W_"+str(n_layers-i)].T)
        #print("Computing gradients grad_Z"+str(i+2)+" with dimensions :"+str(grad_Z[str(i+2)].shape))
    
    gradient = {
        "W_"+str(n_layers):np.dot(act_l["A_"+str(n_layers)].T,grad_Z["1"])/m
    }
    for i in range(1,n_layers):
        gradient["W_"+str(n_layers-i)] = np.dot(act_l["A_"+str(n_layers-i)].T,grad_Z[str(i+1)])/m
    gradient["W_0"] = np.dot(data["x"].T,grad_Z[str(n_layers+1)])

    gradient["B_0"] = np.sum(grad_Z[str(n_layers+1)],axis=1,keepdims=True)
    for i in range(1,n_layers+1):
        gradient["B_"+str(i)] = np.sum(grad_Z[str(n_layers-i+1)],axis=1,keepdims=True)
    
    return gradient


def update_parameters(parameter,grad,n_layers,alpha,reg_fac,m):  #implementing Gradient Descent
    #v_t_w = np.zeros(shape = (weight.shape))
    #v_t_b = np.zeros(shape)
    updated_parameter = {}
    for i in range(n_layers+1):
        updated_parameter["W_"+str(i)] = (1-reg_fac)*parameter["W_"+str(i)] - alpha*grad["W_"+str(i)]
        updated_parameter["B_"+str(i)] = (1-reg_fac)*parameter["B_"+str(i)] - alpha*grad["B_"+str(i)]
    #v_t_w += gradient["weights"]
    #v_t_b += gradient["bias"]
    
    return updated_parameter


epoch = 0
batch_size = 25000
alpha = 0.3 #Learning rate
lambd = 0.0000001 #Regularisation parameter
hidden_layers = 2
n_arr_nodes = [10,10]

print("Generating test data")
train_data = generate_data(t)
#dev_data = generate_data(500000)
parameter_train = random_weights_initializer(hidden_layers,n_arr_nodes)
scaler = StandardScaler()#Function from sklearn to make the standard deviation of the matrix as 1

train_data["x"] = scaler.fit_transform(subtract_mean(train_data["x"]))

while True:
    epoch += 1
    print("Start of epoch number : "+str(epoch))
    for j in range(int(t/batch_size)): 
        index_1 = j*batch_size
        index_2 = (j+1)*batch_size
        
        mini_batch={
            "x":train_data["x"][index_1:index_2],
            "y":train_data["y"][index_1:index_2]
        }
        print("Start of mini batch: "+str(j))
        mini_cache = forward_propagation(mini_batch,parameter_train,hidden_layers)
        mini_grad = back_propagation(mini_batch,mini_cache,parameter_train,hidden_layers,batch_size)
        parameter_train = update_parameters(parameter_train,mini_grad,hidden_layers,alpha,lambd,batch_size) 
        loss = cost_function(mini_cache["out"],mini_batch["y"],batch_size)
        #print("Mini batch :")
        #print(mini_batch)
        #print("mini cache:")
        #print(mini_cache)
        #print("mini grad: ")
        #print(mini_grad)
        print("Loss at this iteration is :"+ str(loss))
        print("Accuracy :"+str(accuracy(mini_cache["out"],mini_batch["y"],batch_size)))
        
    

    if loss <= 1:
        print("Training succesfully complete, Congratulations!!")
        break
print(parameter_train)