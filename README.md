<img src="./itransformer.png" width="400px"></img>

## iTransformer

Implementation of <a href="https://arxiv.org/abs/2310.06625">iTransformer</a> - SOTA Time Series Forecasting using Attention networks, out of Tsinghua / Ant group

All that remains is tabular data (xgboost still champion here) before one can truly declare "Attention is all you need"

In before Apple gets the authors to change the name.

## Install

```bash
$ pip install iTransformer
```

## Usage

```python
import torch
from iTransformer import iTransformer

# using solar energy settings

model = iTransformer(
    num_variates = 137,
    lookback_len = 96,               # or the lookback length in the paper
    dim = 256,                       # model dimensions
    depth = 6,                       # depth
    heads = 8,                       # attention heads
    dim_head = 64,                   # head dimension
    pred_length = (12, 24, 36, 48)   # can be one prediction, or many
)

time_series = torch.randn(2, 96, 137)  # (batch, lookback len, variates)

preds = model(time_series)

# preds -> Dict[int, Tensor[batch, variate, pred_length]]
#       -> (12: (2, 12, 137), 24: (2, 24, 137), 36: (2, 36, 137), 48: (2, 48, 137))
```

## Todo

- [ ] beef up the transformer with latest findings
- [ ] improvise a 2d version

## Citation

```bibtex
@misc{liu2023itransformer,
  title   = {iTransformer: Inverted Transformers Are Effective for Time Series Forecasting}, 
  author  = {Yong Liu and Tengge Hu and Haoran Zhang and Haixu Wu and Shiyu Wang and Lintao Ma and Mingsheng Long},
  year    = {2023},
  eprint  = {2310.06625},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```
