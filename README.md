# CLIP-for-Mushrooms

### About
Finetuning OpenAI's CLIP to match images of mushrooms/fungi with their species/genus name. 
- [Colab Notebook version here](https://drive.google.com/file/d/1l5GS4_hnMvd9W4-JbqCGx3RZPeynCnh1/view?usp=sharing)


### Dataset
- The full dataset (`mushroom_images/`) can be downloaded from [Google Drive](https://drive.google.com/file/d/1RfjX5nEGJNoTEVqaThxumBlm3-75IJR1/view?usp=sharing).

- [Repository for downloading from scratch](https://github.com/pmorris2012/download-mushroomobserver)

- Thank you to [mushroomobserver.org](mushroomobserver.org) for making this data available. 


### Setup and Use
- Install docker with GPU support, then modify and run `docker_train.sh`
- If not using Docker, install [PyTorch and Torchvision](https://pytorch.org/) and the [HuggingFace Transformers](https://huggingface.co/transformers/) libraries with GPU support.
