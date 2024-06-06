
* Create environment & install dependencies
```sh
conda create -n languagebind -y python=3.10
conda activate languagebind

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

* Download weights
```sh
```