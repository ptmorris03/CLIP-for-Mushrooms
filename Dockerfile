FROM huggingface/transformers-pytorch-gpu:latest

COPY requirements.txt /app/requirements.txt

RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python3 -m pip install -r /app/requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
