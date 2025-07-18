import torch
import torch.profiler
from src.main import start_training
from pathlib import Path
from datetime import datetime
import time


def profile_training():
    """Run training with PyTorch profiler"""
    # Record start time
    print(torch.version.cuda)
    start_time = time.time()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        start_training()

    # Calculate total execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Check if profiler captured any events
    if not prof.events():
        print("No profiler events captured! Training may have finished too early.")
        return

    key_avg = prof.key_averages(group_by_stack_n=3)

    # Generate timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd()

    # Create results content with execution time header
    execution_header = f"EXECUTION TIME: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)\n"
    execution_header += (
        f"Profiling completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    execution_header += "=" * 80 + "\n\n"

    cuda_results = f"TOP 15 OPERATIONS BY CUDA TIME:\n{'=' * 60}\n"
    cuda_results += key_avg.table(sort_by="cuda_time_total", row_limit=100)

    cpu_results = f"\nTOP 15 OPERATIONS BY CPU TIME:\n{'=' * 60}\n"
    cpu_results += key_avg.table(sort_by="cpu_time_total", row_limit=100)

    # Save to file with execution time at the top
    profile_file = output_dir / f"profiler_results_{timestamp}.txt"
    with open(profile_file, "w") as f:
        f.write(execution_header)
        f.write(cuda_results)
        f.write(cpu_results)

    # Print file path and execution time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Profiler results saved to: {profile_file}")


if __name__ == "__main__":
    profile_training()
