import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

import time
import os
import argparse

def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, device

def cleanup_ddp():
    dist.destroy_process_group()

def print_rank0(message):
    if dist.get_rank()==0:
        print(message)

def main():
    rank, local_rank, world_size, device = setup_ddp()

    print_rank0(f'Using {world_size} GPUs for distributed training')
    print_rank0(f'Device mapping: Rank {rank} -> GPU {local_rank}')

    batch_size = 64  
    learning_rate = 0.001
    num_epochs = 10

    print_rank0(f'Global batch size: {batch_size * world_size}')
    print_rank0(f'Per gpu batch size: {batch_size}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    os.makedirs('./data', exist_ok=True)

    if rank==0:
        print("Download MNIST dataset")
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    dist.barrier()

    if rank!=0:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    print_rank0(f"Train samples: {len(train_dataset)}")
    print_rank0(f"Test samples: {len(test_dataset)}")
    print_rank0(f"Train batches per GPU: {len(train_loader)}")
    print_rank0(f"Test batches per GPU: {len(test_loader)}")

    model = CNN().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print_rank0("\nStarting distributed training...")
    print_rank0("=" * 60)

    total_start_time = time.time()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)

        print_rank0(f'\nEpoch {epoch+1}/{num_epochs}')
        print_rank0('-' * 50)
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, rank)
        test_loss, test_acc = test_model(model, test_loader, criterion, device, world_size)

        if rank == 0:
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f} seconds')

    
    if rank == 0:
        total_training_time = time.time() - total_start_time
        
        print("\n" + "=" * 60)
        print("DISTRIBUTED TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"Average Time per Epoch: {total_training_time/num_epochs:.2f} seconds")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Speedup with {world_size} GPUs: ~{world_size:.1f}x")
        
        model_path = 'cnn_mnist_model_ddp.pth'
        torch.save(model.module.state_dict(), model_path)  
        print(f"Model saved as '{model_path}'")
        
        save_training_history(train_losses, train_accuracies, test_losses, test_accuracies)
        benchmark_inference_ddp(model, test_loader, device)
    
    cleanup_ddp()
    if rank==0:
        return train_losses, train_accuracies, test_losses, test_accuracies
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = F.relu(self.conv3(x))             
        x = self.pool(x)                      
        x = self.dropout1(x)
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device, rank):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    batch_start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
        if batch_idx % 200 == 0 and rank == 0:
            batch_time = time.time() - batch_start_time
            print(f'  Rank {rank}: Batch {batch_idx:4d}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Time: {batch_time:.2f}s')
            batch_start_time = time.time()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    loss_tensor = torch.tensor(epoch_loss).to(device)
    acc_tensor = torch.tensor(epoch_acc).to(device)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item(), acc_tensor.item()
    
def test_model(model, test_loader, criterion, device, world_size):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    
    loss_tensor = torch.tensor(test_loss).to(device)
    acc_tensor = torch.tensor(test_acc).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    global_acc = acc_tensor.item() / world_size    
    return loss_tensor.item(), global_acc

def count_parameters(model):
    if hasattr(model, 'module'):
        return sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_history(train_losses, train_accuracies, test_losses, test_accuracies):
    history_file = 'training_history_ddp.txt'
    with open(history_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tTest_Loss\tTest_Acc\n")
        for i, (tl, ta, tel, tea) in enumerate(zip(train_losses, train_accuracies, test_losses, test_accuracies)):
            f.write(f"{i+1}\t{tl:.4f}\t{ta:.2f}\t{tel:.4f}\t{tea:.2f}\n")
    print(f"Training history saved to '{history_file}'")

def benchmark_inference_ddp(model, test_loader, device, num_batches=10):
    model.eval()
    
    print(f"\nBenchmarking inference speed on {num_batches} batches...")
    
    with torch.no_grad():
        total_time = 0
        total_samples = 0
        
        for i, (data, target) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            data = data.to(device, non_blocking=True)
            torch.cuda.synchronize()
            start_time = time.time()
            
            output = model(data)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += data.size(0)
        
        avg_time_per_batch = total_time / num_batches
        avg_time_per_sample = total_time / total_samples
        
        print(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
        print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")
        print(f"Throughput: {1/avg_time_per_sample:.0f} samples/second")

if __name__ == "__main__":
    print("Distributed CNN MNIST Training Script")
    script_start_time = time.time()
    result = main()
    if result is not None:
        total_script_time = time.time() - script_start_time
        print(f"\nTotal script execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)")