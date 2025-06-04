## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode),for beam search decode.
- For these who failed install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try `cd ctcdecode` and `pip install .`
- You can install other required modules by conducting
  `pip install -r requirements.txt`
  `pip install transformers`

## Data Preparation

1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
3. CSL dataset Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/datasets/2021_csl_daily/)

Download datasets and extract them, no further data preprocessing needed.

# SLR

### Weights

Here we provide the performance of the model and its corresponding weights.

| Dataset    | Backbone    | Dev WER | Test WER | Pretrained model |
| ---------- | ----------- | ------- | -------- | ---------------- |
| Phoenix14T | SlowFast101 |   16.7  |   18.2   | [[Google Drive]](https://huggingface.co/datasets/xhonghu/M2ETP/tree/main/phoenix2014-t)  |
| Phoenix14  | SlowFast101 |   17.6  |   17.3   | [[Google Drive]](https://huggingface.co/datasets/xhonghu/M2ETP/tree/main/phoenix2014)  |
| CSL-Daily  | SlowFast101 |   24.8  |   23.9   | [[Google Drive]](https://huggingface.co/datasets/xhonghu/M2ETP/tree/main/csl-daily)  |

### Evaluate

To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py --load-weights path_to_weight.pt --phase test`

### Training

Before you start training, download the pre-trained SlowFast pkl file by running the following code:

```bash
mkdir ckpt && cd ckpt
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl
```

<br>

To Training the M2ETP model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py `

Multi-machine training (In fact, the results of the Multi-machine run are not good):

`python -m torch.distributed.launch --nproc_per_node=2 main.py --device 0,1`


### Acknowledgments

Our code is based on [SlowFastSign](https://github.com/kaistmm/SlowFastSign) and [TwoStream](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork).
