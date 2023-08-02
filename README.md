# PopGo
This is the pytorch version (pytorch 1.9.1) of the implementation of the paper: "PopGo: Reducing Popularity Bias in Collaborative Filtering"
Our data and code for data splitting will be released soon.

Four files containing codes:  
data.py, parse.py, model.py , main.py 

To run the code, First run:
```python setup.py build_ext --inplace```
to install tools used in evaluation

# Run code

```python main.py --modeltype  MACRMF```

 
Change "MACRMF" to the model you want :

BPRMF, BCEMF, IPSMF, MACRMF, CausEMF, PopGoMF, LGN, IPSLGN


