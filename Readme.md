## Environment Setup

Python version 3.7.16

torch version 1.12.0+cu102

```bash
conda create --name ga2e python=3.7
conda activate ga2e
pip install -r requirements.txt
```

## Running Code

### [Link prediction](https://github.com/shoyua/GA2E/tree/eca5a3853e4075d50ee815b4b7acbcd70c17ead0/link_prediction)


```bash
#parameters
#data optional[cora|citeseer|pubmed|coaphoto|coacomputer|coacs|coaphysics]
python pretrain.py --data cora
```

### Graph Classification

```bash
python main.py --data imdb-binary
```
