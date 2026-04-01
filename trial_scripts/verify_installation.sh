#!/bin/bash
docker run -v /home/battistini/doc-to-lora:/workspace --rm immagine_prova:latest bash -c "
echo '=== Environment Check Inside Container ===';

# Python
echo -n 'Python: ';
python3 --version 2>/dev/null || echo 'NOT INSTALLED';

# PyTorch
echo -n 'PyTorch: ';
python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED';

# CUDA visible to PyTorch
echo -n 'CUDA (PyTorch sees): ';
python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'NOT INSTALLED';

# CXX11 ABI
echo -n 'CXX11 ABI (PyTorch): ';
python3 -c 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)' 2>/dev/null || echo 'NOT INSTALLED';

# flash-attn
echo -n 'flash-attn: ';
python3 -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'NOT INSTALLED';

# flashinfer
echo -n 'flashinfer-python: ';
python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null || echo 'NOT INSTALLED';

echo '=== Check Complete ===';
"