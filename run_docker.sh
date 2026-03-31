#!/bin/bash

docker run -v /home/battistini/doc_to_lora:/workspace --rm --gpus all immagine_prova:latest \
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"