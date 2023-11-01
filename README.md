# PopGo
This is the pytorch version (pytorch 1.9.1) of the implementation of the paper: "Robust Collaborative Filtering to Popularity Distribution Shift"
Our data and code for data splitting will be released soon.

Four files containing codes:  
data.py, parse.py, model.py , main.py 

To run the code, First run:
```python setup.py build_ext --inplace```
to install tools used in evaluation

# Run code

```python main.py --modeltype  MACRMF```

 
Change "MACRMF" to the model you want:

BPRMF, BCEMF, IPSMF, MACRMF, CausEMF, PopGoMF, LGN, IPSLGN

# Framework

![Alt text](framework.pdf)


# Citation
```bash
@article{PopGo,
  author       = {An Zhang and
                  Wenchang Ma and
                  Jingnan Zheng and
                  Xiang Wang and
                  Tat{-}Seng Chua},
  title        = {Robust Collaborative Filtering to Popularity Distribution Shift},
  journal      = {TOIS},
  year         = {2023}
}
```

