# Nonlinear synthesis and error budgets

At $[q,p]=2i$, define $A=q^3$ and $B=\mathrm{Weyl}(q^2p)$.
Their commutator is $[A,B]=6iq^4$. An ordered group commutator obeys

$$e^{-itB}e^{-isA}e^{itB}e^{isA}
=e^{6istq^4+O(s^2t,st^2)}.$$

The compiler chooses $s=\sqrt{|h|/6}$ and
$t=\mathrm{sign}(h)\sqrt{|h|/6}$ for a quartic pulse $e^{ihq^4}$.
It realizes B through the cubic identity

$$B=\frac{(q+p)^3-(q-p)^3-2p^3}{6}.$$

These powers denote operator products; expansion yields Weyl ordering. Symmetric
splitting of the B terms preserves inverse consistency. Rotated quadratures
$x_\theta=q\cos\theta+p\sin\theta$ provide the corresponding rotated quartics.

For Kerr, the Weyl-ordered identity at hbar=2 is

$$n^2=\frac{x_0^4+x_{\pi/2}^4+x_{\pi/4}^4+x_{-\pi/4}^4}{24}
-\frac{q^2+p^2}{4}.$$

The constant from converting $q^2p^2+p^2q^2$ to Weyl order cancels the constant
in $(q^2+p^2-2)^2/16$. Omitting this ordering step gives the wrong phase generator.
Repeated product-formula slices approximate the sum of these terms.

The expansion is useful on energy-controlled states; unbounded oscillator
operators prevent a uniform whole-Hilbert-space norm claim from a chosen step
count. Tests compare finite-energy states to independent direct matrix evolution
and refine steps. Numerical Fock projection and physical finite-energy injection
introduce additional errors, tested independently.

The general target gate set follows the [CV universality construction](https://arxiv.org/abs/quant-ph/9810082).
The software front end currently accepts quadrature powers of degree 1–4 and
Kerr, not arbitrary-degree polynomial expressions or black-box unitary synthesis.

See the [practical compilation guide](../user-guide/universal-compilation.md).
