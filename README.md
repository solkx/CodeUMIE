# code-umie


## 1. Datasets
The M3D dataset and its videos can be obtained from the [official library](https://github.com/solkx/m3d), place the downloaded dataset in the `/data/m3d` directory and the images in the `/img` directory.

## 2. Pretrained Model and Feature
2.1 Download the [qwen3-8b](https://huggingface.co/Qwen/Qwen3-8B) model and place it in the `/pretrain/qwen3-8b` directory.

2.2 Download visual pre-trained model [AL](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/AL_LLaMA_2_7B_Finetuned.pth?download=true), [VL](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/VL_LLaMA_2_7B_Finetuned.pth?download=true) and [imagebind](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/resolve/main/imagebind_huge.pth?download=true) to the `/pretrain/` directory. 

2.3 Download [visual feature](https://drive.google.com/file/d/117UcJJ-BWjL9TAIcrJEuqsFzB7FCJFBC/view?usp=drive_link) to the `/feature/` directory. 


## 3. Train and Inference
3.1 Run `train.py` for training and testing.

3.2 Run `result.py` to obtain performance metrics on the test set.
