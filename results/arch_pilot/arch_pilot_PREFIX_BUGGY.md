# Architecture pilot

Selection on validation C-index (mean over the two pilot datasets, 2 seeds each), tie-broken by validation IBS.


## transformer (Cox family)

| arch                |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------------|---------:|---------:|----------:|----------:|
| transformer_h64_l2  |   0.5301 |   0.2713 |    0.5008 |    0.2995 |
| transformer_h128_l4 |   0.5295 |   0.3104 |    0.5414 |    0.3442 |
| transformer_h128_l2 |   0.4847 |   0.3379 |    0.5042 |    0.3742 |


**Selected: `transformer_h64_l2`** (val CI 0.5301, val IBS 0.2713)


Per-dataset breakdown:

| arch                | dataset   |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------------|:----------|---------:|---------:|----------:|----------:|
| transformer_h128_l2 | nasa      |   0.4694 |   0.5916 |    0.5084 |    0.6595 |
| transformer_h128_l2 | scania    |   0.5    |   0.0843 |    0.5    |    0.0889 |
| transformer_h128_l4 | nasa      |   0.5567 |   0.5359 |    0.5836 |    0.5994 |
| transformer_h128_l4 | scania    |   0.5022 |   0.0849 |    0.4991 |    0.0889 |
| transformer_h64_l2  | nasa      |   0.5618 |   0.4569 |    0.5025 |    0.5099 |
| transformer_h64_l2  | scania    |   0.4984 |   0.0857 |    0.4992 |    0.0891 |



## gru_attn (DDH family)

| arch          |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------|---------:|---------:|----------:|----------:|
| gru_attn_h64  |   0.7719 |   0.1751 |    0.7704 |    0.1766 |
| gru_attn_h128 |   0.7613 |   0.2883 |    0.7621 |    0.3202 |


**Selected: `gru_attn_h64`** (val CI 0.7719, val IBS 0.1751)


Per-dataset breakdown:

| arch          | dataset   |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------|:----------|---------:|---------:|----------:|----------:|
| gru_attn_h128 | nasa      |   0.5724 |   0.5048 |    0.5809 |    0.5629 |
| gru_attn_h128 | scania    |   0.9502 |   0.0718 |    0.9433 |    0.0774 |
| gru_attn_h64  | nasa      |   0.6547 |   0.1509 |    0.6424 |    0.1565 |
| gru_attn_h64  | scania    |   0.8891 |   0.1992 |    0.8984 |    0.1968 |

