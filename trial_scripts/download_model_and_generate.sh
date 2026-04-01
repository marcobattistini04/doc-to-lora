#!/bin/bash

IMAGE_NAME="immagine_prova:latest"

docker run --rm \
--gpus all \
    -v "/home/battistini/doc-to-lora/data:/workspace/data" \
    -v "/home/battistini/doc-to-lora/trained_d2l:/workspace/trained_d2l" \
    -v "/home/battistini/doc-to-lora/ctx_to_lora:/workspace/ctx_to_lora" \
    -e PYTHONPATH="/workspace" \
    $IMAGE_NAME \
    python3 - <<'PYTHON_CODE'
import torch
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)


with open("data/sakana_wiki.txt", "r") as f:
    doc = f.read()

chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


model.internalize(doc)


outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
PYTHON_CODE