import random
import torch
import json

from nltk_ut import tokenize, stemming, bagOfWords 
from model import NeuralNet

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("todo.json",'r') as file:
    intents=json.load(file)

File="data.pth"
data=torch.load(File)

input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
allWords=data["allWords"]
tags=data["tags"]
model_state=data["model_state"]

#model=NeuralNet(input_size,hidden_size,output_size).to(device)
model=NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

bot_name="Bot"
print("Let's chat! Type 'quit' to exit")
while True:
    sentence=input('You:')
    if sentence=="quit":
        break

    sentence=tokenize(sentence)
    X=bagOfWords(sentence,allWords)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]

    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]

    if prob.item()>0.75:
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print(f'{bot_name}:{random.choice(intent["responses"])}')

    else:
        print(f"{bot_name}:I don't understand. Please try again")

