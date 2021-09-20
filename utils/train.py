import torch
import numpy as np
import matplotlib.pyplot as plt

def train(model, optimizer, criterion, train_loader, valid_loader, num_of_epoch, model_name):
    total_step = len(train_loader)
    min_valid_loss = np.inf
    device = torch.device("cuda")
    train_losses = []
    valid_losses = []

    for epoch in range(num_of_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device).float()
            labels = torch.as_tensor(labels['age']).to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        for i, (imgs, labels) in enumerate(valid_loader):
            with torch.no_grad():
                imgs = imgs.to(device).float()
                labels = torch.as_tensor(labels['age']).to(device)
            
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)    
        
        avg_valid_loss = valid_loss/len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        print(f"Epoch: {epoch+1}/{num_of_epoch}, Train Loss: {avg_train_loss},  Validation Loss: {avg_valid_loss}")
        
        
        if min_valid_loss > avg_valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss} ---> {avg_valid_loss})')
            min_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), model_name)
            
    plt.plot(list(range(1, num_of_epoch+1)), train_losses)
    plt.plot(list(range(1, num_of_epoch+1)), valid_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Loss')
    plt.show()