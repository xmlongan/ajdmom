# ajdmom

## Description

The `ajdmom` package is a **Python library designed for the automatic derivation of
moment formulas for well-established Affine Jump Diffusion (AJD) processes.** 
It significantly enhances the usability of AJD models by providing **explicit, 
closed-form expressions for unconditional moments and conditional moments,
up to any positive integer order.**


Beyond just moments, `ajdmom` offers a valuable tool for **sensitivity analysis** 
by computing the partial derivatives of these moments with respect to model 
parameters. The package features a **modular architecture**, facilitating easy 
adaptation and extension by researchers. `ajdmom` is open-source and readily 
available for installation from GitHub or the Python Package Index (PyPI).

The moments derived by `ajdmom` have **broad applications** in quantitative finance 
and stochastic modeling, including:

- **Density Approximation**: Accurately approximating unknown probability densities 
  (e.g., through Pearson distributions) by matching derived moments. This enables 
  efficient European option pricing under the concerned models.

- **Exact Simulation**: Facilitating the exact simulation of AJD models in an 
  efficient way when compared to characteristic function inversion methods.

- **Parameter Estimation**: Formulating explicit moment estimators for AJD models 
  whose likelihood functions are not analytically solvable.

Consequently, `ajdmom` has the potential to become an essential instrument for 
researchers and practitioners demanding comprehensive AJD model analysis.

### Supported Models & Moment Types

| Model | Unconditional Moments | Conditional Moments - I | Conditional Moments - II |
|:-----:|:---------------------:|:-----------------------:|:------------------------:|
| Heston|           ✅          |           ✔️            |           N/A           |
| 1FSVJ |           ✅          |           ✔️            |           N/A           |
| 2FSV  |           ✅          |           ✔️            |           N/A           |
| 2FSVJ |           ✅          |           ✔️            |           N/A           |
| SRJD  |           ✅          |            ✅           |            ✅           |
| SVVJ  |           ✅          |            ✅           |            ✅           |
| SVCJ  |           ✅          |            ✅           |            ✅           |
| SVIJ  |          ✔️           |           ✔️            |            ✅           |

Notes: 

- ✅ **Implemented:** The feature is fully implemented.
- ✔️ **Applicable:** The feature is applicable to this model but not yet implemented. 
- **N/A Not Applicable:** The feature is not relevant or applicable for this model. 
- **Unconditional Moments:** Include raw moments ($E[y_n^l]$), 
  central moments ($E[\bar{y}_n^l]$), and autocovariances n 
  ($cov(y_n^{l_1},y_{n+1}^{l_2})$). 
  - *Note: Autocovariances are not yet available for SRJD and SVCJ.*
- **Conditional Moments - I:** Derivation where the initial state of the variance 
  process ($v_0$) is given.
- **Conditional moments - II:** Derivation where both the initial state ($v_0$) and 
  the realized jump times and jump sizes in the variance process over the concerned
  interval are given beforehand.

## Simple Usage

To get the formula for the first moment $\mathbb{E}[y_n]$ for the Heston Stochastic
Volatility (SV) model ( $y_n$ denotes the return over the nth interval of length $h$ ), 
run the following code snippet:

``` python
from ajdmom import mdl_1fsv # mdl_1fsv -> mdl_1fsvj, mdl_2fsv, mdl_2fsvj
from pprint import pprint

m1 = mdl_1fsv.moment_y(1)   # 1 in moment_y(1) -> 2,3,4...

# moment_y() -> cmoment_y()             : central moment
# dpoly(m1, wrt), wrt = 'k','theta',... : partial derivative

msg = "which is a Poly with attribute keyfor = \n{}"
print("moment_y(1) = "); pprint(m1); print(msg.format(m1.keyfor))
```

which produces:

```         
moment_y(1) = 
{(0, 1, 0, 0, 1, 0, 0, 0): Fraction(-1, 2),
 (0, 1, 0, 1, 0, 0, 0, 0): Fraction(1, 1)}
which is a Poly with attribute keyfor = 
('e^{-kh}', 'h', 'k^{-}', 'mu', 'theta', 'sigma_v', 'rho', 'sqrt(1-rho^2)')
```

Within the produced results, the two key-value pairs, namely (0,1,0,0,1,0,0,0): Fraction(-1,2) and 
(0,1,0,1,0,0,0,0): Fraction(1,1), correspond to the following expressions:

$$
-\frac{1}{2}\times e^{-0kh}h^1k^{-0}\mu^0\theta^1\sigma_v^0\rho^0\left(\sqrt{1-\rho^2}\right)^0,
$$

$$
1\times e^{-0kh}h^1k^{-0}\mu^1\theta^0\sigma_v^0\rho^0\left(\sqrt{1-\rho^2}\right)^0,
$$

respectively. The summation of these terms reproduces the first moment of the Heston
SV model: $\mathbb{E}[y_n] = (\mu-\theta/2)h$. This demonstrates that the `ajdmom` 
package successfully encapsulates the model's dynamics into a computationally 
manipulable form, specifically leveraging a custom dictionary data structure, 
referred to as `Poly`, to encode the moment's expression.

## Documentation

The documentation is hosted on <http://www.yyschools.com/ajdmom/>

## Ongoing Development

This code is being developed on an on-going basis at the author's [Github site](https://github.com/xmlongan/ajdmom).

## Support

For support in using this software, submit an [issue](https://github.com/xmlongan/ajdmom/issues/new).
