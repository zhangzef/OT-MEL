<h1 align="center"> Optimal Transport Guided Correlation Assignment for Multimodal Entity Linking </h1>

<p align="center"> 
  <a href="https://arxiv.org/pdf/2406.01934" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-FFFFFF?&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU%2BYXJYaXY8L3RpdGxlPjxwYXRoIGQ9Ik0zLjg0MjMgMGExLjAwMzcgMS4wMDM3IDAgMCAwLS45MjIuNjA3OGMtLjE1MzYuMzY4Ny0uMDQzOC42Mjc1LjI5MzggMS4xMTEzbDYuOTE4NSA4LjM1OTctMS4wMjIzIDEuMTA1OGExLjAzOTMgMS4wMzkzIDAgMCAwIC4wMDMgMS40MjI5bDEuMjI5MiAxLjMxMzUtNS40MzkxIDYuNDQ0NGMtLjI4MDMuMjk5LS40NTM4LjgyMy0uMjk3MSAxLjE5ODZhMS4wMjUzIDEuMDI1MyAwIDAgMCAuOTU4NS42MzUuOTEzMy45MTMzIDAgMCAwIC42ODkxLS4zNDA1bDUuNzgzLTYuMTI2IDcuNDkwMiA4LjAwNTFhLjg1MjcuODUyNyAwIDAgMCAuNjgzNS4yNTk3Ljk1NzUuOTU3NSAwIDAgMCAuODc3Ny0uNjEzOGMuMTU3Ny0uMzc3LS4wMTctLjc1MDItLjMwNi0xLjE0MDdsLTcuMDUxOC04LjM0MTggMS4wNjMyLTEuMTNhLjk2MjYuOTYyNiAwIDAgMCAuMDA4OS0xLjMxNjVMNC42MzM2LjQ2MzlzLS4zNzMzLS40NTM1LS43NjgtLjQ2M3ptMCAuMjcyaC4wMTY2Yy4yMTc5LjAwNTIuNDg3NC4yNzE1LjU2NDQuMzYzOWwuMDA1LjAwNi4wMDUyLjAwNTUgMTAuMTY5IDEwLjk5MDVhLjY5MTUuNjkxNSAwIDAgMS0uMDA3Mi45NDVsLTEuMDY2NiAxLjEzMy0xLjQ5ODItMS43NzI0LTguNTk5NC0xMC4zOWMtLjMyODYtLjQ3Mi0uMzUyLS42MTgzLS4yNTkyLS44NDFhLjczMDcuNzMwNyAwIDAgMSAuNjcwNC0uNDQwMVptMTQuMzQxIDEuNTcwMWEuODc3Ljg3NyAwIDAgMC0uNjU1NC4yNDE4bC01LjY5NjIgNi4xNTg0IDEuNjk0NCAxLjgzMTkgNS4zMDg5LTYuNTEzOGMuMzI1MS0uNDMzNS40NzktLjY2MDMuMzI0Ny0xLjAyOTJhMS4xMjA1IDEuMTIwNSAwIDAgMC0uOTc2My0uNjg5em0tNy42NTU3IDEyLjI4MjMgMS4zMTg2IDEuNDEzNS01Ljc4NjQgNi4xMjk1YS42NDk0LjY0OTQgMCAwIDEtLjQ5NTkuMjYuNzUxNi43NTE2IDAgMCAxLS43MDYtLjQ2NjljLS4xMTE5LS4yNjgyLjAzNTktLjY4NjQuMjQ0Mi0uOTA4M2wuMDA1MS0uMDA1NS4wMDQ3LS4wMDU1eiIvPjwvc3ZnPg%3D%3D"> 
  </a>
  <a href="#step-2-download-the-data">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-FFFFFF?&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tISBGb250IEF3ZXNvbWUgUHJvIDYuNC4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlIChDb21tZXJjaWFsIExpY2Vuc2UpIENvcHlyaWdodCAyMDIzIEZvbnRpY29ucywgSW5jLiAtLT48cGF0aCBkPSJNNDQ4IDgwdjQ4YzAgNDQuMi0xMDAuMyA4MC0yMjQgODBTMCAxNzIuMiAwIDEyOFY4MEMwIDM1LjggMTAwLjMgMCAyMjQgMFM0NDggMzUuOCA0NDggODB6TTM5My4yIDIxNC43YzIwLjgtNy40IDM5LjktMTYuOSA1NC44LTI4LjZWMjg4YzAgNDQuMi0xMDAuMyA4MC0yMjQgODBTMCAzMzIuMiAwIDI4OFYxODYuMWMxNC45IDExLjggMzQgMjEuMiA1NC44IDI4LjZDOTkuNyAyMzAuNyAxNTkuNSAyNDAgMjI0IDI0MHMxMjQuMy05LjMgMTY5LjItMjUuM3pNMCAzNDYuMWMxNC45IDExLjggMzQgMjEuMiA1NC44IDI4LjZDOTkuNyAzOTAuNyAxNTkuNSA0MDAgMjI0IDQwMHMxMjQuMy05LjMgMTY5LjItMjUuM2MyMC44LTcuNCAzOS45LTE2LjkgNTQuOC0yOC42VjQzMmMwIDQ0LjItMTAwLjMgODAtMjI0IDgwUzAgNDc2LjIgMCA0MzJWMzQ2LjF6Ii8%2BPC9zdmc%2B"> 
  </a>
</p>

<p align="center"> This repository is the official implementation for the paper "Optimal Transport Guided Correlation Assignment for Multimodal Entity Linking" published in Findings of ACL 2024.  </p>

<p align="center">
  <img src="model.jpg" alt="OT-MEL" width="1080">
</p>


## Usage

### Step 1: Set up the environment

We recommend using Conda to manage virtual environments, and we use Python version 3.8.12.
```bash
conda create -n otmel python==3.8.12
conda activate otmel
pip install -r requirements.txt
```

Please install the specified versions of Python libraries according to the requirements.txt file.

Note that the versions of PyTorch, Transformers, and PyTorch Lightning may have a slight impact on the results. To fully reproduce the results of the paper, we recommend installing the specified versions.

<div id="dataset"></div>

### Step 2: Download the data

You may download WikiMEL and RichpediaMEL from https://github.com/seukgcode/MELBench and WikiDiverse from https://github.com/wangxw5/wikiDiverse.

Or download our cleaned data [WikiMEL](https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/ETtT1zwqdDdAmE-uxHMX5EAB7bCGb1Eh2AuafB0tijDdyg?e=IT9E8a), [RichpediaMEL](https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/ERikbOQuoWFHrA_AizcuCbgB8PBOiRqCV4U0lZfxUN-6kg?e=speIdh), [WikiDiverse](https://mailustceducn-my.sharepoint.com/:u:/g/personal/pfluo_mail_ustc_edu_cn/EQgQKn4VeghChY_lhUoyBIMBKz6aTS00DFKOL1dqxP_bEg?e=yRpKkU) (Password: kdd2023).


### Step 3: Modify the data path

Please modify the configuration files under the "config" directory (including the YAML files for all 3 datasets) and replace `YOUR_PATH` in the `data` field of each configuration file with the path to your corresponding dataset.

**NOTE: Due to the uploaded training files of RichpediaMEL, mention images are stored in the folder `mention_images`. You need to modify the `mention_img_folder` in the `richpediamel.yaml` config file or rename the `mention_images` folder to `mention_image`.** (Thank [Zhiwei Hu](https://github.com/zhiweihu1103) for bringing up this [issue](https://github.com/pengfei-luo/MIMIC/issues/2))

### Step 4: Start the training

Now you can execute `bash run.sh <gpu_id> <dataset_name>` to begin the training.
```bash
bash run.sh 0 wikimel       # for WikiMEL
bash run.sh 0 richpediamel  # for RichpediaMEL
bash run.sh 0 wikidiverse   # for WikiDiverse
```

## Code Structure
The code is organized as follows:
```text
├── codes
│   ├── main.py
│   ├── model
│   │   ├── lightning_ot.py
│   │   └── modeling_ot.py
│   └── utils
│       ├── dataset.py
│       └── functions.py
├── config
│   ├── richpediamel.yaml
│   ├── wikidiverse.yaml
│   └── wikimel.yaml
├── readme.md
├── requirements.txt
└── run.sh
```

## Acknowledgements
We acknowledge the outstanding open-source contributions from [MIMIC](https://github.com/pengfei-luo/MIMIC) and [UniOT-for-UniDA](https://github.com/changwxx/UniOT-for-UniDA).


## Citation
If you find this project useful in your research, please cite the following paper:
```bibtex
@article{zhang2024optimal,
  title={Optimal Transport Guided Correlation Assignment for Multimodal Entity Linking},
  author={Zhang, Zefeng and Sheng, Jiawei and Zhang, Chuang and Liang, Yunzhi and Zhang, Wenyuan and Wang, Siqi and Liu, Tingwen},
  journal={arXiv preprint arXiv:2406.01934},
  year={2024}
}
```


## Contact Information

If you have any questions, please contact zhangzef1999@163.com.