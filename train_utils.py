import time
import random
import torch
from torch import nn
import torchaudio.functional as F



### Data Augmentation ###
class RandomEffector(nn.Module):
    def __init__(self, sample_rate, p=0.5, lowpass_cutoff=300):
        super(RandomEffector, self).__init__()
        self.sample_rate = sample_rate
        self.gain = F.gain
        self.lowpass = F.lowpass_biquad
        self.lowpass_cutoff = lowpass_cutoff
        self.prob = p
    
    def forward(self, signal):
        # Applies each effect at p probabilty
        if random.random() < self.prob:
            signal = self.gain(signal)
        if random.random() < self.prob:
            signal = self.lowpass(signal, self.sample_rate, self.lowpass_cutoff)
        return signal


class FixedSignalLength(nn.Module):
    def __init__(self, sample_rate, max_ms):
        super(FixedSignalLength, self).__init__()
        self.max_len = sample_rate//1000 * max_ms

    def forward(self, signal):
        num_channel, sig_len = signal.shape
        if (sig_len > self.max_len):
            signal = signal[:, :self.max_len]
        elif (sig_len < self.max_len):
            pad_head_len = random.randint(0, self.max_len - sig_len)
            pad_tail_len = self.max_len - sig_len - pad_head_len
            pad_head = torch.zeros((num_channel, pad_head_len))
            pad_tail = torch.zeros((num_channel, pad_tail_len))
            signal = torch.cat((pad_head, signal, pad_tail), dim=1)
        return signal


### Train Function ###
def train_and_save(model, loss_func, optimizer, scheduler, data_loaders, end_epoch, save_path, load_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", device)
    model.to(device)

    # Initialize records
    train_loader, test_loader = data_loaders
    batch_size = train_loader.batch_size
    start_epoch = 0
    train_loss, test_loss = torch.zeros(end_epoch-start_epoch), torch.zeros(end_epoch-start_epoch)
    train_acc, test_acc = torch.zeros(end_epoch-start_epoch), torch.zeros(end_epoch-start_epoch)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        cp_train_loss, cp_test_loss = checkpoint["loss"]
        cp_train_acc, cp_test_acc = checkpoint["acc"]
        train_loss = torch.cat([cp_train_loss, train_loss])
        test_loss = torch.cat([cp_test_loss, test_loss])
        train_acc = torch.cat([cp_train_acc, train_acc])
        test_acc = torch.cat([cp_test_acc, test_acc])
        print("Load path successful")

    # Begin training
    start_time = time.time()
    num_train_batch = len(train_loader)
    num_test_batch = len(test_loader)
    record_interval = num_train_batch//4
    print("\nBegin training from epoch {} to {}...\n".format(start_epoch+1, end_epoch))
    for i in range(start_epoch, end_epoch):
        print(f"Epoch {i+1}/{end_epoch}")

        # Train mode
        model.train()
        batch_loss = 0.
        batch_correct = 0.
        for j, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_correct += (outputs.data.argmax(axis=1) == labels.argmax(axis=1)).sum().item()
            batch_loss += loss.item()
            if j%record_interval == 0:
                print(f"Batch {j+1} completed...")

        train_loss[i] = batch_loss / num_train_batch
        train_acc[i] = batch_correct / (num_train_batch * batch_size)
        
        # Evaluate mode
        print("Evaluating...")
        model.eval()
        batch_loss = 0.
        batch_correct = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
            batch_correct += (outputs.data.argmax(axis=1) == labels.argmax(axis=1)).sum().item()
            batch_loss += loss.item()
            
        test_loss[i] = batch_loss / num_test_batch
        test_acc[i] = batch_correct / (num_test_batch * batch_size)
        
        print(f"Train Loss: {train_loss[i]:.2f} | Train Accuracy: {train_acc[i]:.2f}")
        print(f"Test Loss: {test_loss[i]:.2f} | Test Accuracy: {test_acc[i]:.2f}")
        print(f"Elapsed time: {time.time()-start_time:.1f}s\n")

    print(f"Training ended ({time.time()-start_time:.1f}s), saving...")

    # Save results
    torch.save({
        "epoch": end_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": (train_loss, test_loss),
        "acc": (train_acc, test_acc),
    }, save_path)
    print("Save successful")