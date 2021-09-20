import torch


def evaluation(model, test_loader):
    device = torch.device("cuda")
    loss_per_class = torch.zeros([102], dtype=torch.long).to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        error = torch.zeros(0).to(device)
        for imgs, labels in test_loader:
            imgs = imgs.to(device).float()
            labels = torch.as_tensor(labels['age']).to(device)
            outputs = model(imgs)
            
            _, pred = torch.max(outputs.data, 1)
            
            for i in range(pred.size(0)):
                loss_per_class[labels[i]] +=  abs(labels[i]-pred[i])
                        
            error = torch.cat([error, torch.abs(
                torch.subtract(torch.reshape(labels, (-1,)), torch.reshape(pred, (-1,)))
            )])
                        
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
    #print(f"Accuracy: {(100*correct)/total}%")
    print(f"Mean Absolute Error: {(torch.mean(error))}")
    print(f"Minimum: {torch.min(error)}, Maximum: {torch.max(error)}, Median: {torch.median(error)}")
    
    return loss_per_class
