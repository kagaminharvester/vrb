#!/bin/bash
###############################################################################
# GPU Profiling Script for VR Body Segmentation
#
# This script runs various GPU profiling tools and collects performance metrics.
# Usage: ./scripts/profile_gpu.sh [options]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROFILE_TYPE="all"
OUTPUT_DIR="profiling_results"
DURATION=10
BATCH_SIZE=4

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Profile type: nvprof, nsight, pytorch, all (default: all)"
    echo "  -o, --output DIR       Output directory for results (default: profiling_results)"
    echo "  -d, --duration SEC     Profiling duration in seconds (default: 10)"
    echo "  -b, --batch-size N     Batch size for testing (default: 4)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t pytorch                     # Run PyTorch profiler only"
    echo "  $0 -t nsight -d 30                # Run Nsight for 30 seconds"
    echo "  $0 --batch-size 8 -o my_results   # Profile with batch size 8"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            PROFILE_TYPE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════"
echo "  VR Body Segmentation - GPU Profiler"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA not available?${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print system info
echo -e "${BLUE}System Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
echo ""

# Print configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Profile Type:  $PROFILE_TYPE"
echo "  Duration:      ${DURATION}s"
echo "  Batch Size:    $BATCH_SIZE"
echo "  Output Dir:    $OUTPUT_DIR"
echo ""

###############################################################################
# PyTorch Profiler
###############################################################################
run_pytorch_profiler() {
    echo -e "${YELLOW}Running PyTorch Profiler...${NC}"

    cat > /tmp/profile_pytorch.py << 'EOF'
import torch
import torch.nn as nn
import time
import sys

# Get parameters
duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
output_dir = sys.argv[3] if len(sys.argv) > 3 else "profiling_results"

print(f"PyTorch Profiling: {duration}s, batch_size={batch_size}")

# Create simple model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 1, 3, padding=1),
    nn.Sigmoid()
).cuda()

model.eval()

# Input tensor
input_tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')

# Warmup
print("Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = model(input_tensor)
torch.cuda.synchronize()

# Profile with PyTorch profiler
print("Profiling...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    start_time = time.time()
    iterations = 0

    while time.time() - start_time < duration:
        with torch.no_grad():
            _ = model(input_tensor)
        iterations += 1

    torch.cuda.synchronize()

elapsed = time.time() - start_time
fps = (iterations * batch_size) / elapsed

print(f"\nResults:")
print(f"  Iterations: {iterations}")
print(f"  FPS: {fps:.2f}")
print(f"  Time per iteration: {(elapsed/iterations)*1000:.2f}ms")

# Export results
print(f"\nExporting profiling results to {output_dir}/pytorch_trace.json")
prof.export_chrome_trace(f"{output_dir}/pytorch_trace.json")

# Print summary
print("\nTop 10 GPU operations by time:")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\nTop 10 operations by memory:")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

# Save summary to file
with open(f"{output_dir}/pytorch_summary.txt", "w") as f:
    f.write("GPU Operations (by time):\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write("\n\nMemory Usage:\n")
    f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

print(f"\nResults saved to {output_dir}/")
EOF

    python /tmp/profile_pytorch.py $DURATION $BATCH_SIZE $OUTPUT_DIR

    echo -e "${GREEN}✓ PyTorch profiling complete${NC}"
    echo ""
}

###############################################################################
# NVIDIA Nsight Systems
###############################################################################
run_nsight_systems() {
    echo -e "${YELLOW}Running NVIDIA Nsight Systems...${NC}"

    if ! command -v nsys &> /dev/null; then
        echo -e "${YELLOW}Warning: nsys not found, skipping Nsight Systems profiling${NC}"
        echo "Install from: https://developer.nvidia.com/nsight-systems"
        return
    fi

    # Create profiling script
    cat > /tmp/profile_nsight.py << 'EOF'
import torch
import torch.nn as nn
import time
import sys

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

print(f"Nsight profiling: {duration}s, batch_size={batch_size}")

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 1, 3, padding=1),
).cuda()

input_tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(input_tensor)

torch.cuda.synchronize()

# Profile
start_time = time.time()
while time.time() - start_time < duration:
    with torch.no_grad():
        _ = model(input_tensor)

torch.cuda.synchronize()
EOF

    nsys profile \
        --output="$OUTPUT_DIR/nsight_report" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        python /tmp/profile_nsight.py $DURATION $BATCH_SIZE

    echo -e "${GREEN}✓ Nsight Systems profiling complete${NC}"
    echo -e "${BLUE}  View report: nsys-ui $OUTPUT_DIR/nsight_report.qdrep${NC}"
    echo ""
}

###############################################################################
# NVIDIA Nsight Compute
###############################################################################
run_nsight_compute() {
    echo -e "${YELLOW}Running NVIDIA Nsight Compute...${NC}"

    if ! command -v ncu &> /dev/null; then
        echo -e "${YELLOW}Warning: ncu not found, skipping Nsight Compute profiling${NC}"
        echo "Install from: https://developer.nvidia.com/nsight-compute"
        return
    fi

    # Create profiling script
    cat > /tmp/profile_ncu.py << 'EOF'
import torch
import torch.nn as nn

batch_size = 4

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 1, 3, padding=1),
).cuda()

input_tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')

# Run a few iterations
for _ in range(5):
    with torch.no_grad():
        _ = model(input_tensor)

torch.cuda.synchronize()
EOF

    ncu \
        --output="$OUTPUT_DIR/ncu_report" \
        --force-overwrite \
        --set full \
        python /tmp/profile_ncu.py

    echo -e "${GREEN}✓ Nsight Compute profiling complete${NC}"
    echo -e "${BLUE}  View report: ncu-ui $OUTPUT_DIR/ncu_report.ncu-rep${NC}"
    echo ""
}

###############################################################################
# Memory Profiling
###############################################################################
run_memory_profiler() {
    echo -e "${YELLOW}Running GPU Memory Profiler...${NC}"

    cat > /tmp/profile_memory.py << 'EOF'
import torch
import torch.nn as nn
import time
import sys

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4
output_dir = sys.argv[3] if len(sys.argv) > 3 else "profiling_results"

print(f"Memory profiling: {duration}s, batch_size={batch_size}")

# Reset stats
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 1, 3, padding=1),
).cuda()

input_tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')

# Profile memory usage
memory_samples = []

start_time = time.time()
iterations = 0

while time.time() - start_time < duration:
    with torch.no_grad():
        _ = model(input_tensor)

    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved() / (1024**2)

    memory_samples.append({
        'iteration': iterations,
        'allocated': allocated,
        'reserved': reserved
    })

    iterations += 1

torch.cuda.synchronize()

# Get final stats
peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)

print(f"\nMemory Statistics:")
print(f"  Peak Allocated: {peak_allocated:.2f} MB")
print(f"  Peak Reserved:  {peak_reserved:.2f} MB")
print(f"  Iterations:     {iterations}")

# Save results
import json
with open(f"{output_dir}/memory_profile.json", "w") as f:
    json.dump({
        'peak_allocated_mb': peak_allocated,
        'peak_reserved_mb': peak_reserved,
        'samples': memory_samples
    }, f, indent=2)

print(f"\nMemory profile saved to {output_dir}/memory_profile.json")
EOF

    python /tmp/profile_memory.py $DURATION $BATCH_SIZE $OUTPUT_DIR

    echo -e "${GREEN}✓ Memory profiling complete${NC}"
    echo ""
}

###############################################################################
# GPU Utilization Monitoring
###############################################################################
run_gpu_utilization() {
    echo -e "${YELLOW}Monitoring GPU Utilization...${NC}"

    # Start workload in background
    cat > /tmp/profile_workload.py << 'EOF'
import torch
import torch.nn as nn
import time
import sys

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 1, 3, padding=1),
).cuda()

input_tensor = torch.rand(batch_size, 3, 640, 640, device='cuda')

start_time = time.time()
while time.time() - start_time < duration:
    with torch.no_grad():
        _ = model(input_tensor)
EOF

    # Run workload and monitor
    python /tmp/profile_workload.py $DURATION $BATCH_SIZE &
    WORKLOAD_PID=$!

    # Monitor GPU
    nvidia-smi dmon -s umct -c $(($DURATION * 2)) -o TD > "$OUTPUT_DIR/gpu_utilization.log" &
    MONITOR_PID=$!

    # Wait for workload to complete
    wait $WORKLOAD_PID
    sleep 1
    kill $MONITOR_PID 2>/dev/null || true

    echo -e "${GREEN}✓ GPU utilization monitoring complete${NC}"
    echo -e "${BLUE}  Log saved to: $OUTPUT_DIR/gpu_utilization.log${NC}"
    echo ""
}

###############################################################################
# Main execution
###############################################################################

case $PROFILE_TYPE in
    pytorch)
        run_pytorch_profiler
        run_memory_profiler
        ;;
    nsight)
        run_nsight_systems
        run_nsight_compute
        ;;
    utilization)
        run_gpu_utilization
        ;;
    all)
        run_pytorch_profiler
        run_memory_profiler
        run_gpu_utilization
        run_nsight_systems
        run_nsight_compute
        ;;
    *)
        echo -e "${RED}Error: Invalid profile type '$PROFILE_TYPE'${NC}"
        exit 1
        ;;
esac

# Generate summary report
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Profiling Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Available files:"
ls -lh "$OUTPUT_DIR/" | tail -n +2
echo ""
echo -e "${GREEN}✓ Profiling complete!${NC}"
echo ""
echo "Next steps:"
echo "  - View PyTorch trace: chrome://tracing (load pytorch_trace.json)"
echo "  - View Nsight Systems: nsys-ui $OUTPUT_DIR/nsight_report.qdrep"
echo "  - View Nsight Compute: ncu-ui $OUTPUT_DIR/ncu_report.ncu-rep"
echo "  - View memory profile: cat $OUTPUT_DIR/memory_profile.json"
echo ""
