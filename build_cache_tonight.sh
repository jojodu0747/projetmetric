#!/bin/bash

# Script to build Gram cache for all 24 configurations (4 backbones × 6 layers)
# Run this overnight to pre-compute all Gram matrices

echo "============================================================"
echo "BUILDING GRAM CACHE FOR ALL CONFIGURATIONS"
echo "============================================================"
echo "Start time: $(date)"
echo ""
echo "Configurations:"
echo "  - 4 backbones: sd_vae, dinov2_vitb14, vgg19, lpips_vgg"
echo "  - 6 layers per backbone = 24 total configs"
echo "  - 100 reference images"
echo "  - 20 evaluation images × ~38 degradation levels"
echo ""
echo "This will take several hours. Output will be saved to:"
echo "  results/cache/gram_cache.pkl"
echo "============================================================"
echo ""

# Activate environment if needed (uncomment and adjust if using conda/venv)
# source activate your_env
# or
# conda activate your_env

# Run the cache builder
python build_gram_cache.py 2>&1 | tee build_cache_log.txt

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "CACHE BUILD COMPLETED SUCCESSFULLY"
    echo "End time: $(date)"
    echo "Cache file: results/cache/gram_cache.pkl"
    echo "Log file: build_cache_log.txt"
    echo "============================================================"
    echo ""
    echo "To run evaluation tomorrow, use:"
    echo "  python run_from_gram_cache.py"
else
    echo ""
    echo "============================================================"
    echo "ERROR: Cache build failed!"
    echo "End time: $(date)"
    echo "Check build_cache_log.txt for details"
    echo "============================================================"
    exit 1
fi
