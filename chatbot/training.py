import json
import numpy as np
from nltk_ut import tokenize, stemming, bagOfWords 

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open("todo.json",'r') as file:
    intents=json.load(file)
   
allWords=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        i=tokenize(pattern)
        allWords.extend(i)
        xy.append((i,tag))

excludeWords=['!','.',',','?']
allWords=[stemming(i) for i in allWords if i not in excludeWords]
allWords=sorted(set(allWords))
tags=sorted(set(tags))
#print(allWords)
#print(f'tags:{tags}')


Xtrain=[]
Ytrain=[]
for(patternSentence,tag) in xy:
    bag=bagOfWords(patternSentence,allWords)
    Xtrain.append(bag)

    label=tags.index(tag)
    Ytrain.append(label)

Xtrain=np.array(Xtrain)
Ytrain=np.array(Ytrain)

class ChatBotDataset(Dataset):

    def __init__(self):
        self.n_samples=len(Xtrain)
        self.Xdata=Xtrain
        self.Ydata=Ytrain

    def __getitem__(self,index):
        return self.Xdata[index],self.Ydata[index]

    def __len__(self):
        return self.n_samples


batch_size=8
input_size=len(allWords)
hidden_size=8
output_size=len(tags)
learning_rate=0.001
num_epochs=1000

#print(len(Xtrain[0]), len(allWords))
#print(tags,output_size)

dataset=ChatBotDataset()

trainLoader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model=NeuralNet(input_size,hidden_size,output_size).to(device)
model=NeuralNet(input_size,hidden_size,output_size)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in trainLoader:
        #words=words.to(device)
        #labels=labels.to(dtype=torch.long).to(device)
        words=words
        labels=labels.to(dtype=torch.long)

        outputs=model(words)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1)%100==0:
        print(f'Epoch[{epoch+1}/{num_epochs}],loss={loss.item():.4f}')
    
print(f'Final loss={loss.item():.4f}')

data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "allWords":allWords,
    "tags":tags

}
File="data.pth"
torch.save(data,File)

print(f'TRAINING COMPLETE. FILE SAVED to {File}')


