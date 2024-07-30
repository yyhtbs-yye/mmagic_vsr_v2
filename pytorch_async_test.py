import torch
import time

# Check if CUDA is available
device_sync = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
device_async = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Define the size of the tensors and number of iterations
tensor_size = (101, 101)
num_iterations = 100

start_time = time.time()
for _ in range(num_iterations):
    x = torch.randn(tensor_size, device=device_sync)
    y = torch.randn(tensor_size, device=device_sync)
    z = torch.matmul(x, y).to(device_async)
    # No synchronize here, allowing for asynchronous execution
torch.cuda.synchronize()  # Wait for all operations to complete
async_duration = time.time() - start_time
print(f"First Asynchronous execution time: {async_duration:.4f} seconds")

del x, y, z
torch.cuda.empty_cache()

running_sync_duration = 0
running_async_duration = 0

N = 100

for i in range(N):
    # Synchronous execution example
    start_time = time.time()
    for _ in range(num_iterations):
        x = torch.randn(tensor_size, device=device_sync)
        y = torch.randn(tensor_size, device=device_sync)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()  # Ensure the operation is complete before continuing
    sync_duration = time.time() - start_time
    running_sync_duration += sync_duration
    
    del x, y, z
    torch.cuda.empty_cache()

    start_time = time.time()
    for _ in range(num_iterations):
        x = torch.randn(tensor_size, device=device_async)
        y = torch.randn(tensor_size, device=device_async)
        z = torch.matmul(x, y)
        # No synchronize here, allowing for asynchronous execution
    torch.cuda.synchronize()  # Wait for all operations to complete
    async_duration = time.time() - start_time
    running_async_duration += async_duration
    
    del x, y, z
    torch.cuda.empty_cache()

print(f"Synchronous execution time: {running_sync_duration/N:.4f} seconds")
print(f"Second Asynchronous execution time: {running_async_duration/N:.4f} seconds")