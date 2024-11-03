## Run DrugBAN on Our Data to Reproduce Results

To train PepBAN, where we provide the basic configurations for all hyperparameters in `config.py`. For different in-domain and cross-domain tasks, the customized task configurations can be found in respective `CDAN.yaml` files.
 
```
$ python main.py --cfg "CDAN.yaml" --data ${dataset}
```
