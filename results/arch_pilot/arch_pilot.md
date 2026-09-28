# Architecture pilot

Selection on validation C-index (mean over the two pilot datasets, 2 seeds each), tie-broken by validation IBS.


## transformer (Cox family)

| arch                |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------------|---------:|---------:|----------:|----------:|
| transformer_h64_l2  |   0.5431 |   0.2976 |    0.5094 |    0.3281 |
| transformer_h128_l4 |   0.5077 |   0.3092 |    0.5033 |    0.3437 |
| transformer_h128_l2 |   0.4491 |   0.4395 |    0.4873 |    0.4704 |


**Selected: `transformer_h64_l2`** (val CI 0.5431, val IBS 0.2976)


Per-dataset breakdown:

| arch                | dataset   |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------------|:----------|---------:|---------:|----------:|----------:|
| transformer_h128_l2 | nasa      |   0.4012 |   0.5351 |    0.4656 |    0.5987 |
| transformer_h128_l2 | scania    |   0.4971 |   0.3439 |    0.5091 |    0.3421 |
| transformer_h128_l4 | nasa      |   0.5155 |   0.5318 |    0.5066 |    0.5958 |
| transformer_h128_l4 | scania    |   0.5    |   0.0866 |    0.5    |    0.0915 |
| transformer_h64_l2  | nasa      |   0.5861 |   0.5095 |    0.5191 |    0.566  |
| transformer_h64_l2  | scania    |   0.5    |   0.0857 |    0.4997 |    0.0902 |



## gru_attn (DDH family)

| arch          |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------|---------:|---------:|----------:|----------:|
| gru_attn_h128 |   0.5311 |   0.6379 |    0.5211 |    0.6662 |
| gru_attn_h64  |   0.5043 |   0.5272 |    0.5461 |    0.5525 |


**Selected: `gru_attn_h128`** (val CI 0.5311, val IBS 0.6379)


Per-dataset breakdown:

| arch          | dataset   |   val_ci |   val_bs |   test_ci |   test_bs |
|:--------------|:----------|---------:|---------:|----------:|----------:|
| gru_attn_h128 | nasa      |   0.5622 |   0.5452 |    0.5421 |    0.6099 |
| gru_attn_h128 | scania    |   0.5    |   0.7307 |    0.5    |    0.7225 |
| gru_attn_h64  | nasa      |   0.5078 |   0.493  |    0.5851 |    0.5511 |
| gru_attn_h64  | scania    |   0.5007 |   0.5614 |    0.5071 |    0.5539 |

