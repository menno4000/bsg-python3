## Bayesian Skip Gram (Python 3 capable implementation)

This repository contains a fork of the [Bayesian Skip Gram](https://github.com/abrazinskas/BSG) implementation ready for use with Python 3 on Windows Systems.

### Encoding Notice

Encoding Functions have been altered from the Original Implementation to enable training on Windows.

On Unix Systems, adapt encodings from `'latin-1'` to `'utf-8'`


### Usage

To train the model, run the `run_bsg.py` script:
```
python run_bsg_invoice.py --epochs=1 --alpha=0.005 --max_vocab_size=100000 --batch_size=500 --nr_neg_samples=10 --embedding_size=100
```

To start the inference api, supply the `bsg_api.py` script with the trained model parent directory (in project root \output\ by default):
```
python bsg_api.py --model_index=0  # trained model output at \output\0\
```

Then get nearest neighbours of words from the trained embedding by calling `localhost:5000`:
```
curl --request POST 'http://localhost:5000/nearest_neighbours?query=interesting word series'
```



