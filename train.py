
import model
import torch
import torchvision
import torch.optim as optim
import utilities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle



bce = torch.nn.BCELoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')



# Hyperparameters
n_epochs = 120
n_layers = 4
batch_size = 64
learning_rate = 0.001


#######################################################

# Load MNIST Data
train_loader = DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])), batch_size=batch_size, shuffle=True, pin_memory=False)

test_loader = DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])), batch_size=batch_size, shuffle=True, pin_memory=False)


################################################
dev = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"

#Create Model and Optimizer
model = model.DTP(n_layers, 784, 240, 10)
model.to(dev)


#################################################

#Create Containers
h = [torch.randn(1, 1, device=dev) for i in range(n_layers + 1)]
h_hat = [torch.randn(1, 1, device=dev) for i in range(n_layers)]
#target = torch.FloatTensor(batch_size, 10)

#################################################


losses = []
accuracies = []
test_losses = []
test_accuracies = []

for ep in range(n_epochs):
    num_correct = 0
    total_batch_loss = 0


    # TRAIN
    for batch_idx, (images, y) in enumerate(train_loader):
        images = images.view(-1, 784).to(dev)
        y = y.to(dev)

        #Transform targets, y, to onehot vector
        target = torch.zeros(images.size(0), 10, device=dev)
        utilities.to_one_hot(target, y)

        #Compute values, outputs from feedforward pass
        model.compute_values(h, images)

        #Compute targets, outputs of decoders using difference target prop
        if ep % 6 == 0 and batch_idx < 5:
            p=True
        else:
            p=False

        model.compute_targets(h_hat, h, target, p=p)

        #print(h[0].shape, h[1].shape, h_hat[0].shape, h[-1].shape, h_hat[-1].shape)
        global_loss = model.train_encoders(h, h_hat)

        #Train feedbacks
        model.train_decoders(h)

        #Add number correct and loss for this batch to totals
        num_correct += utilities.compute_num_correct(h[-1], y)
        total_batch_loss += global_loss

    #Each epoch append accuracy for epoch and avg loss.
    losses.append(total_batch_loss.item() / (batch_idx + 1))
    accuracies.append(num_correct / 60000)
    print(ep, 'avg_loss: ', losses[-1], ' accuracy: ', accuracies[-1])


    #TEST
    with torch.no_grad():
        num_correct = 0
        total_batch_loss = 0
        for batch_idx, (images, y) in enumerate(test_loader):
            images = images.view(-1, 784).to(dev)
            y = y.to(dev)

            # Transform targets, y, to onehot vector
            target = torch.zeros(images.size(0), 10, device=dev)
            utilities.to_one_hot(target, y)

            # Compute feedforward values
            model.compute_values(h, images)

            # Check loss at last layer
            global_loss = torch.mean(bce(h[-1], target).sum(-1))

            # Add number correct and loss for this batch to totals
            num_correct += utilities.compute_num_correct(h[-1], y)
            total_batch_loss += global_loss

        # Each epoch append accuracy for epoch and avg loss.
        test_losses.append(total_batch_loss.item() / (batch_idx + 1))
        test_accuracies.append(num_correct / 10000)
        print(ep, 'test_avg_loss: ', test_losses[-1], ' test_accuracy: ', test_accuracies[-1])



###################################

#Plot
plt.plot(losses, label='train_loss')
plt.plot(test_losses, label='test_loss')
plt.legend()
plt.show()

plt.plot(accuracies, label='train_acc')
plt.plot(test_accuracies, label='test_acc')
plt.legend()
plt.show()


#Write data to file
with open('trainLoss.data', 'wb') as filehandle:
    pickle.dump(losses, filehandle)

with open('trainAcc.data', 'wb') as filehandle:
    pickle.dump(accuracies, filehandle)

with open('testLoss.data', 'wb') as filehandle:
    pickle.dump(test_losses, filehandle)

with open('testAcc.data', 'wb') as filehandle:
    pickle.dump(test_accuracies, filehandle)






