######### Part 1 installation and import ##########

from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset

import torch
from io import open
import glob
import os
import numpy as np
import unicodedata
import string
import random
import torch.nn as nn
import time
import math
import syft as sy
import pandas as pd
import random
from syft.frameworks.torch.federated import utils

from syft.workers import WebsocketClientWorker
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

print('installation suceeded')
# check by here if import is correct

############### part 2 data prepocessing ################

#Load all the files in a certain path
def findFiles(path):
    return glob.glob(path)

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

#convert a string 's' in unicode format to ASCII format
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#dictionary containing the nation as key and the names as values
#Example: category_lines["italian"] = ["Abandonato","Abatangelo","Abatantuono",...]
category_lines = {}
#List containing the different categories in the data
all_categories = []

for filename in findFiles('data/names/*.txt'):
    print(filename)
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines   
    
n_categories = len(all_categories)  

class LanguageDataset(Dataset):
    # Constructor
    def __init__(self, text, labels, transform=None):
        self.data = text
        self.targets = labels # categories
        #self.to_torchtensor()
        self.transform = transform
    
    def to_torchtensor(self):
        self.data = torch.from_numpy(self.text, requires_grad=True)
        self.labels = torch.from_numpy(self.targets, requires_grad=True)
  
    # Returns length of dataset/batches
    def __len__(self):
        return len(self.data)
  
    # Returns data and target[torch tensor ]
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)
      
        return sample, target
      
# Arguments for the program
class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.learning_rate = 0.005
        self.epochs = 10000
        self.federate_after_n_batches =15000
        self.seed = 1
        self.print_every = 200
        self.plot_every = 100
        self.use_cuda = False
    
args = Arguments()

names_list = []
#Set of labels (Y)
category_list = []

#Convert into a list with corresponding label.

for nation, names in category_lines.items():
    #iterate over every single name
    for name in names:
        names_list.append(name)      #input data point
        category_list.append(nation) #label
 categories_numerical = pd.factorize(category_list)[0]

# Categories with tensor
category_tensor = torch.tensor(np.array(categories_numerical), dtype=torch.long)

categories_numpy = np.array(category_tensor)

def letterToIndex(letter):
    return all_letters.find(letter)
    

# Turn a line into a <line_length x 1 x n_letters>
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor    
    
    
# Turn a list of strings into a list of tensors
def list_strings_to_list_tensors(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensor(line)
        lineNumpy = lineTensor.numpy()
        lines_tensors.append(lineNumpy)
        
    return(lines_tensors)

lines_tensors = list_strings_to_list_tensors(names_list)

ax_line_size = max(len(x) for x in lines_tensors)

def lineToTensorFillEmpty(line, max_line_size):
    tensor = torch.zeros(max_line_size, 1, n_letters) #notice the difference between this method and the previous one
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
        
        #Vectors with (0,0,.... ,0) are placed where there are no characters
    return tensor

def list_strings_to_list_tensors_fill_empty(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensorFillEmpty(line, max_line_size)
        lines_tensors.append(lineTensor)
    return(lines_tensors)

lines_tensors = list_strings_to_list_tensors_fill_empty(names_list)

array_lines_tensors = np.stack(lines_tensors)
#However, such operation introduces one extra dimension (look at the dimension with index=2 having size '1')
print(array_lines_tensors.shape)
#Because that dimension just has size 1, we can get rid of it with the following function call
array_lines_proper_dimension = np.squeeze(array_lines_tensors, axis=2)
print(array_lines_proper_dimension.shape)

def find_start_index_per_category(category_list):
    categories_start_index = {}
    
    #Initialize every category with an empty list
    for category in all_categories:
        categories_start_index[category] = []
    
    #Insert the start index of each category into the dictionary categories_start_index
    #Example: "Italian" --> 203
    #         "Spanish" --> 19776
    last_category = None
    i = 0
    for name in names_list:
        cur_category = category_list[i]
        if(cur_category != last_category):
            categories_start_index[cur_category] = i
            last_category = cur_category
        
        i = i + 1
        
    return(categories_start_index)

categories_start_index = find_start_index_per_category(category_list)

def randomChoice(l):
    rand_value = random.randint(0, len(l) - 1)
    return l[rand_value], rand_value


def randomTrainingIndex():
    category, rand_cat_index = randomChoice(all_categories) #cat = category, it's not a random animal
    #rand_line_index is a relative index for a data point within the random category rand_cat_index
    line, rand_line_index = randomChoice(category_lines[category])
    category_start_index = categories_start_index[category]
    absolute_index = category_start_index + rand_line_index
    return(absolute_index)
  
########## part 3 remote workers set-up and dataset distribution ##############

hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
#alice = sy.VirtualWorker(hook, id="alice")  
#bob = sy.VirtualWorker(hook, id="bob")  

#If you have your workers operating remotely, like on Raspberry PIs
kwargs_websocket_alice = {"host": "6", "hook": hook}
alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket_alice)
kwargs_websocket_bob = {"host": "ip_bob", "hook": hook}
bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket_bob)
workers_virtual = [alice, bob]

langDataset =  LanguageDataset(array_lines_proper_dimension, categories_numpy)

#assign the data points and the corresponding categories to workers.
print("assignment starts")
federated_train_loader = sy.FederatedDataLoader(
            langDataset.federate(workers_virtual),
            batch_size=args.batch_size)
print("assignment completed") 
# time test for this part shows that time taken is not significant for this stage

print("Generating list of batches for the workers...")
list_federated_train_loader = list(federated_train_loader)
print("Listed generated")
# time test for this part shows that time taken is significant for this stage
# it is executed here to show that this process is independent of model
# data flow across network can be observed

############ part 4 create RNN and form federated training function ############

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
#Instantiate RNN

device = torch.device("cuda" if args.use_cuda else "cpu")
model = RNN(n_letters, n_hidden, n_categories).to(device)
#The final softmax layer will produce a probability for each one of our 18 categories
print(model)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def fed_avg_every_n_iters(model_pointers, iter, federate_after_n_batches):
        models_local = {}
        
        if(iter % args.federate_after_n_batches == 0):
            for worker_name, model_pointer in model_pointers.items():
#                #need to assign the model to the worker it belongs to.
                models_local[worker_name] = model_pointer.copy().get()
            model_avg = utils.federated_avg(models_local)
           
            for worker in workers_virtual:
                model_copied_avg = model_avg.copy()
                model_ptr = model_copied_avg.send(worker) 
                model_pointers[worker.id] = model_ptr
                
        return(model_pointers)     

def fw_bw_pass_model(model_pointers, line_single, category_single):
    #get the right initialized model
    model_ptr = model_pointers[line_single.location.id]   
    line_reshaped = line_single.reshape(max_line_size, 1, len(all_letters))  ######## place break point here to inspect instructions
    line_reshaped, category_single = line_reshaped.to(device), category_single.to(device)
    #Firstly, initialize hidden layer
    hidden_init = model_ptr.initHidden() 
    #And now zero grad the model
    model_ptr.zero_grad()
    hidden_ptr = hidden_init.send(line_single.location)
    amount_lines_non_zero = len(torch.nonzero(line_reshaped.copy().get()))
    #now need to perform forward passes
    for i in range(amount_lines_non_zero): 
        output, hidden_ptr = model_ptr(line_reshaped[i], hidden_ptr) 
    criterion = nn.NLLLoss()   
    loss = criterion(output, category_single) 
    loss.backward()
    
    model_got = model_ptr.get() 
    
    #Perform model weights' updates    
    for param in model_got.parameters():
        param.data.add_(-args.learning_rate, param.grad.data)
        
        
    model_sent = model_got.send(line_single.location.id)
    model_pointers[line_single.location.id] = model_sent
    
    return(model_pointers, loss, output)
            
  
    
def train_RNN(n_iters, print_every, plot_every, federate_after_n_batches, list_federated_train_loader):
    current_loss = 0
    all_losses = []    
    
    model_pointers = {}
    
    #Send the initialized model to every single worker just before the training procedure starts
    for worker in workers_virtual:
        model_copied = model.copy()
        model_ptr = model_copied.send(worker) ######## place break point at this line to monitor network data flow
        model_pointers[worker.id] = model_ptr

    #extract a random element from the list and perform training on it
    for iter in range(1, n_iters + 1):        
        random_index = randomTrainingIndex()
        line_single, category_single = list_federated_train_loader[random_index]
        #print(category_single.copy().get())
        line_name = names_list[random_index]
        model_pointers, loss, output = fw_bw_pass_model(model_pointers, line_single, category_single)
        #model_pointers = fed_avg_every_n_iters(model_pointers, iter, args.federate_after_n_batches)
        #Update the current loss a
        loss_got = loss.get().item() 
        current_loss += loss_got
        
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
             
        if(iter % print_every == 0):
            output_got = output.get()  #Without copy()
            guess, guess_i = categoryFromOutput(output_got)
            category = all_categories[category_single.copy().get().item()]
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss_got, line_name, guess, correct))
    return(all_losses, model_pointers)

######## part 5 perform training and evaluate performance ########

start = time.time()
print("training starts")
all_losses, model_pointers = train_RNN(args.epochs, args.print_every, args.plot_every, args.federate_after_n_batches, list_federated_train_loader)

def predict(model, input_line, worker, n_predictions=3):
    model = model.copy().get()
    print('\n> %s' % input_line)
    model_remote = model.send(worker)
    line_tensor = lineToTensor(input_line)
    line_remote = line_tensor.copy().send(worker)
    #line_tensor = lineToTensor(input_line)
    #output = evaluate(model, line_remote)
    # Get top N categories
    hidden = model_remote.initHidden()
    hidden_remote = hidden.copy().send(worker)
        
    with torch.no_grad():
        for i in range(line_remote.shape[0]):
            output, hidden_remote = model_remote(line_remote[i], hidden_remote)
        
    topv, topi = output.copy().get().topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])
        
 predict(model_pointers["alice"], "Tai", alice) # here test a name didn't appear in the dataset but with 'typical' feature of a language
 predict(model_pointers["alice"], "Tai", alice) # here test a name appeared in the dataset 
