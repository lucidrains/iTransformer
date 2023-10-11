<img src="./itransformer.png" width="400px"></img>

## iTransformer

Implementation of <a href="https://arxiv.org/abs/2310.06625">iTransformer</a> - SOTA Time Series Forecasting using Attention networks, out of Tsinghua / Ant group

All that remains is tabular data (xgboost still champion here) before one can truly declare "Attention is all you need"

In before Apple gets the authors to change the name.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source current artificial intelligence techniques.

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
- [ ] improvise a 2d version - either global pool across time at end, or use a CLS token for attention pooling

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

```bibtex
@misc{burtsev2020memory,
    title   = {Memory Transformer},
    author  = {Mikhail S. Burtsev and Grigory V. Sapunov},
    year    = {2020},
    eprint  = {2006.11527},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{Darcet2023VisionTN,
    title   = {Vision Transformers Need Registers},
    author  = {Timoth'ee Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:263134283}
}
```
