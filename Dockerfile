FROM huggingface/transformers-pytorch-gpu:latest

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install -r /app/requirements.txt

CMD ["/bin/bash"]
