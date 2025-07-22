# Replicating the experiments in the paper

## Demonstration

To replicate the results in [Figure 1](../results/m1-1fsv.png), run this command from the `scripts` directory:

```    
python example1.py
```

To replicate the results in [Figure 2](../results/cov21-1fsv.png), run this command from the `scripts` directory:

```   
python example2.py
```

## Experiments

To replicate the results in [Table 1](../results/comp-mom-1fsvj.png), run this command from the `scripts` directory:

```         
python experiment1.py
```

To replicate the results in [Table 2](../results/comp-cov-1fsvj.png), run this command from the `scripts` directory:

```         
python experiment2.py
```

To replicate the results in [Table 3](../results/comp-cond-mom-svcj.png), run this command from the `scripts` directory:

```         
python experiment3.py
```

To replicate the results in [Table 4](../results/comp-cond2-mom-svcj.png), run this command from the `scripts` directory:

```         
python experiment4.py
```

It should be noted that the results in Tables 1 to 4 may not be
reproduced exactly since the sample moments and covariances used for
comparison are computed based on simulated samples which would change
per simulation.


## Automated Tests

Tests live in the `tests` directory. If you want to run the tests, you first need to make sure package `pytest` has been installed. Then, run this command from the same directory where `pyproject.tomal` is located:

```         
python -m pytest tests/
```

Note that it may take several hours to finish since we are using Euler approximation to generate samples, then compare the derived theoretical moments with their sample counterparts.
