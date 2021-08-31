# Point cloud completion and reconstruction

## Usage:

```
python src/train.py 
  --model graphcnn-ppd
  --mode reconstruction
  --train-dataset /pat/to/dataset 
  --eval-dataset /path/to/dataset
  --multi-resolution true
  --use-max false
```


## Explanation of flags:

| Flag | Description |
| --- | --- |
| model | Choose the model for training and evaluation. Possible values: `pointnet`, `pointetpp`, `dgcnn`, `dgcnn-ppd` |
| mode | Choose the type of task. Possible values: `reconstruction`, `completion`, `completion-missing-regions`|
| train-dataset | Path to train dataset |
| eval-dataset | Path to eval dataset |
| multi-resolution | Enable multi-resolution mode. Available only with dgcnn-ppd |
| use-max | Use the max function in multi-resolution mode |

## Extra

```
python src/train_vae.py
```

Train the variational auto-encoder