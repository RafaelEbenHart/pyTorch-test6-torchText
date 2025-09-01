import torch
from torch import nn
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
from timeit import default_timer as Timer
import ast
from pathlib import Path

# prediction
# torch.manual_seed(42)
def evalModel(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0,0
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for X,y in dataLoader:
            X,y = X.to(device),y.to(device)
            # prediksi
            yPred = model(X)
            # akumulasi loss dan Acc per batch
            loss += lossFn(yPred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=yPred.argmax(dim=1))

        # scale loss and acc to find avg per batch
        loss /= len(dataLoader)
        acc /= len(dataLoader)
    result = {"ModelName" : model.__class__.__name__,# hanya bisa berkerja jika mmodel dibuat dengan class
            "ModelLoss" : loss.item(),
            "ModelAcc" : acc}
    return result

# time
def printTrainTime(start:float,
                   end:float,
                   device: torch.device = None):
    """print perbandingan antara start dan end time"""
    totalTime = end-start
    print(f"Train time on {device}: {totalTime:.3f} seconds")
    return totalTime

# train loop
def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              perBatch: None):
    """Performs training with model trying to learn on dataLoader"""
    model.train()
    trainLoss,trainAcc = 0, 0
    device = next(model.parameters()).device
    for batch, (X,y) in enumerate(dataLoader):
        X,y = X.to(device),y.to(device) # put tu target device
        yPred = model(X) # forward pass
        # Calculate loss and acc
        loss = lossFn(yPred,y)
        trainLoss += loss.item()
        # optimizer zero grad,loss backward,optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy
        yPredClass = torch.argmax(torch.softmax(yPred, dim=1), dim=1)
        trainAcc += (yPredClass == y).sum().item()/len(yPred)
        # show batch
        if perBatch:
            if batch % perBatch == 0:
                print(f"Looked at: {(batch * len(X)) + perBatch}/{len(dataLoader.dataset)} samples ")

    # calculate avg
    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    return trainLoss, trainAcc

# test loop

def testStep(model: torch.nn.Module,
             dataLoader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module):
    """Performs testing with model trying to test on dataLoader"""
    testLoss,testAcc = 0, 0
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for X,y in dataLoader:
            X,y = X.to(device),y.to(device)
            testPred = model(X)
            testLoss += lossFn(testPred,y)
            testPredLabels = testPred.argmax(dim=1)
            testAcc += ((testPredLabels == y).sum().item()/len(testPredLabels))

        testLoss /= len(dataLoader)
        testAcc /= len(dataLoader)
    return testLoss,testAcc

# train and test loop

def train_test_loop (epochs: int,
                     model:torch.nn.Module,
                     lossFn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     train_dataLoader: torch.utils.data.DataLoader,
                     test_dataLoader: torch.utils.data.DataLoader,
                     perBatch: None):
    results = {"train_loss" : [],
              "train_acc" : [],
              "test_loss" : [],
              "test_acc" : []}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           patience=3)
    startTime = Timer()
    for epoch in tqdm(range(epochs), position=0, leave=True):
        train_loss,train_acc = trainStep(model=model,
                                        dataLoader=train_dataLoader,
                                        lossFn=lossFn,
                                        optimizer=optimizer,
                                        perBatch=perBatch)
        test_loss,test_acc = testStep(model=model,
                                      dataLoader=test_dataLoader,
                                      lossFn=lossFn)
        scheduler.step(test_loss)


        tqdm.write(f"\nEpoch:{epoch+1}/{epochs}|"
                   f"Train Loss: {train_loss:.5f}|"
                   f"Train Acc: {train_acc*100:.2f}%|"
                   f"Test Loss: {test_loss:.5f} |"
                   f"Test Acc: {test_acc*100:.2f}%|")
        # memastika data pindah ke cpu dan berubah menjadi tensor float
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)


    endTime = Timer()
    printTrainTime(start=startTime,
                   end=endTime,
                   device=str(next(model.parameters()).device))
    return results

def makePredictions(model:torch.nn.Module,
                    data:list):
    predProbs = []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # prepare the sample (add a batch dimensio and pass to target device)
            sample = torch.unsqueeze(sample,dim=0).to(device)
            # forward pass (model output raw logits)
            predLogits = model(sample)
            # get prediction probability (logit -> preediction probability)
            predProb = torch.softmax(predLogits.squeeze(),dim=0)
            # get predprobs off the CUDA for further calculations
            predProbs.append(predProb.cpu())
    # stack the predProbs to turn list into a tensor
    return torch.stack(predProbs)

# non linear activation
class hyperTanh(nn.Module):
    def forward(self, x) :
        ex = torch.exp(x)
        enx = torch.exp(-x)
        return (ex - enx) / (ex + enx)

# Saving
def Save(path:str,name:str,model:torch.nn.Module):
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    MODEL_NAME = name
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    print(f"Saving your model to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)

# Loading
def load(model: torch.nn.Module, Saved_path:str):
    import torch
    model.load_state_dict(torch.load(f=Saved_path))
    print("loaded")

# save results dict as txt
def save_results_txt( path:str,name:str,results:dict,):
    SAVE_RESULTS = Path(path)
    SAVE_RESULTS.mkdir(parents=True,
                       exist_ok=True)
    SAVE_NAME = name
    SAVE_PATH = SAVE_RESULTS/SAVE_NAME
    with open(SAVE_PATH, "w") as f:
        for key,values in results.items():
            f.write(f"{key}:{values}\n")
    print(f"Saving results to {SAVE_PATH}")

# load result dict as txt
def load_results_txt(SAVE_PATH:str):
    results = {}
    with open(SAVE_PATH) as f:
        for line in f:
            key,values = line.strip().split(":",1)
            results[key] = ast.literal_eval(values)
    print("loaded")
    return results




