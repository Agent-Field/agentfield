#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "AgentField Scale Benchmark Suite"
echo "=============================================="
echo ""

# System info
echo "System Information:"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Arch: $(uname -m)"
echo "  CPUs: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
echo ""

# Go Benchmark
echo "----------------------------------------------"
echo "Running Go SDK Benchmark..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR/go-bench"

# Build
go mod tidy 2>/dev/null || true
go build -o benchmark .

# Run with different scales
echo "Testing scale: 100,000 handlers"
./benchmark --handlers=100000 --iterations=10 --warmup=2 --json > "$RESULTS_DIR/AgentField_Go.json"

echo "Go benchmark complete."
cat "$RESULTS_DIR/AgentField_Go.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Registration: {[r['value'] for r in d['results'] if r['metric']=='registration_time_mean_ms'][0]:.2f}ms\")"

# Python Benchmark
echo ""
echo "----------------------------------------------"
echo "Running Python SDK Benchmark..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR/python-bench"

python3 benchmark.py --handlers=5000 --iterations=10 --warmup=2 --json > "$RESULTS_DIR/AgentField_Python.json"

echo "Python benchmark complete."
cat "$RESULTS_DIR/AgentField_Python.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Registration: {[r['value'] for r in d['results'] if r['metric']=='registration_time_mean_ms'][0]:.2f}ms\")"

# LangChain Benchmark
echo ""
echo "----------------------------------------------"
echo "Running LangChain Baseline..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR/langchain-bench"

if python3 -c "import langchain_core" 2>/dev/null; then
    python3 benchmark.py --tools=1000 --iterations=10 --warmup=2 --json > "$RESULTS_DIR/LangChain_Python.json"
    echo "LangChain benchmark complete."
    cat "$RESULTS_DIR/LangChain_Python.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Registration: {[r['value'] for r in d['results'] if r['metric']=='registration_time_mean_ms'][0]:.2f}ms\")"
else
    echo "  Skipping: langchain-core not installed"
    echo "  Install with: pip install langchain-core"
fi

# CrewAI Benchmark
echo ""
echo "----------------------------------------------"
echo "Running CrewAI Baseline..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR/crewai-bench"

if python3 -c "from crewai.tools import tool" 2>/dev/null; then
    python3 benchmark.py --tools=1000 --iterations=10 --warmup=2 --json > "$RESULTS_DIR/CrewAI_Python.json"
    echo "CrewAI benchmark complete."
    cat "$RESULTS_DIR/CrewAI_Python.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Registration: {[r['value'] for r in d['results'] if r['metric']=='registration_time_mean_ms'][0]:.2f}ms\")"
else
    echo "  Skipping: crewai not installed"
    echo "  Install with: pip install crewai"
fi

# Mastra Benchmark
echo ""
echo "----------------------------------------------"
echo "Running Mastra Baseline..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR/mastra-bench"

if [ -f "package.json" ]; then
    npm install --silent 2>/dev/null
    if npx tsx benchmark.ts --tools=1 --iterations=1 --warmup=0 2>/dev/null; then
        npx tsx benchmark.ts --tools=1000 --iterations=10 --warmup=2 --json > "$RESULTS_DIR/Mastra_TypeScript.json"
        echo "Mastra benchmark complete."
        cat "$RESULTS_DIR/Mastra_TypeScript.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Registration: {[r['value'] for r in d['results'] if r['metric']=='registration_time_mean_ms'][0]:.2f}ms\")"
    else
        echo "  Skipping: @mastra/core not available or failed to run"
        echo "  Install with: npm install @mastra/core"
    fi
else
    echo "  Skipping: package.json not found"
fi

# Generate visualizations
echo ""
echo "----------------------------------------------"
echo "Generating Visualizations..."
echo "----------------------------------------------"
cd "$SCRIPT_DIR"

if python3 -c "import matplotlib, seaborn" 2>/dev/null; then
    python3 analyze.py
else
    echo "  Skipping: matplotlib/seaborn not installed"
    echo "  Install with: pip install matplotlib seaborn"
fi

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
ls -la "$RESULTS_DIR"
