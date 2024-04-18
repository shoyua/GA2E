## Environment Setup

Python version 3.7.16

torch version 1.12.0+cu102

```bash
conda create --name ga2e python=3.7
conda activate ga2e
pip install -r requirements.txt
```

## Running Code

### [Link prediction](link_prediction)


```bash
#parameters
#data optional[cora|citeseer|pubmed|coaphoto|coacomputer|coacs|coaphysics]
python pretrain.py --data cora
```

### [Graph Classification](graph_classification)

```bash
python main.py --data imdb-binary
```
