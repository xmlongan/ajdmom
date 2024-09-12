# ajdmom

## Description

The package `ajdmom` is a `Python` package designed for automatically deriving moment formulas 
for the well-established affine jump diffusion (AJD) processes. `ajdmom` can produce explicit closed-form 
expressions for moments or conditional moments of any order, significantly enhancing the usability 
of AJD models. Additionally, `ajdmom` can compute partial derivatives of these moments with respect to the 
model parameters, offering a valuable tool for sensitivity analysis. The package's modular architecture makes 
it easy for adaptation and extension by researchers. `ajdmom` is open-source and readily available for 
installation from GitHub or the `Python` package index (PyPI).

Currently, `ajdmom` supports computations of moments, central moments, and covariances for the Heston 
Stochastic Volatility (SV) model and its three AJD extensions: 
- SVJ (SV with jumps in the price), 
- Two-Factor SV, 
- Two-Factor SV with jumps in the price. 

Moreover, the package can compute partial 
derivatives of these quantities with respect to model parameters. 

In addition, the package supports 
computations of *conditional* moments and *conditional* central moments for another three AJD extensions:
- SVVJ (SV with jumps in the variance), 
- SVIJ (SV with independent jumps in the price and variance),
- SVCJ (SV with contemporaneous jumps in the price and variance). 

The computations of *conditional* moments and *conditional* central moments is also supported for
- SRJD (Square-Root Jump Diffusion) process.

The moments and covariances obtained through `ajdmom` have far-reaching implications for multiple domains, 
including financial modelling, simulation and parameter estimation. For simulations, these moments can 
establish the underlying probability distributions, leading to significant reductions in computational 
time when contrasted with conventional numerical CF inversion techniques. In parameter estimation, 
the moments serve to formulate explicit moment estimators while the likelihood functions are not analytically 
solvable. Consequently, `ajdmom` has the potential to become an essential instrument for researchers and 
practitioners demanding comprehensive AJD model analysis.

## Simple Usage

To get the formula for the first moment $\mathbb{E}[y_n]$ for the Heston Stochastic Volatility model
( $y_n$ denotes the return over the nth interval of length $h$ ), run the following code snippet:

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

Within the produced results, the two principal key-value pairs, namely (0,1,0,0,1,0,0,0): Fraction(-1,2) and 
(0,1,0,1,0,0,0,0): Fraction(1,1), correspond to the following expressions:

$$
-\frac{1}{2}\times e^{-0kh}h^1k^{-0}\mu^0\theta^1\sigma_v^0\rho^0\left(\sqrt{1-\rho^2}\right)^0,
$$

$$
1\times e^{-0kh}h^1k^{-0}\mu^1\theta^0\sigma_v^0\rho^0\left(\sqrt{1-\rho^2}\right)^0,
$$

respectively. The summation of these terms yields the first moment of the One-Factor SV model: 
$\mathbb{E}[y_n] = (\mu-\theta/2)h$. This demonstrates that the `ajdmom` package successfully encapsulates 
the model's dynamics into a computationally manipulable form, specifically leveraging a custom dictionary 
data structure, referred to as `Poly`, to encode the moment's expression.

## Documentation

The documentation is hosted on <http://www.yyschools.com/ajdmom/>

## Ongoing Development

This code is being developed on an on-going basis at the author's [Github site](https://github.com/xmlongan/ajdmom).

## Support

For support in using this software, submit an [issue](https://github.com/xmlongan/ajdmom/issues/new).
