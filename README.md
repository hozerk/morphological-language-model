# morphological-language-model
Training of transformer based morphological language model from scratch with parallel gpus.

language model can be trained on multiple GPUs with the following script. nproc_per_node parameter is the GPU count
!python -m torch.distributed.launch --nproc_per_node=4 main.py
