# libraries
import numpy as np
from numpy.polynomial import Polynomial
from numpy.linalg import solve
import matplotlib.pyplot as p

# linear ODE solver function
def lode(a, x0):
	N = a.shape[0] - 1
	s = Polynomial(a).roots()
	S = np.zeros([N, N], dtype=complex)
	for i in range(N):
		S[i] = s**(i)
	A = solve(S, x0)
	def xh(t):
		xh = 0.0
		for i in range(N):
			xh += A[i]*np.exp(s[i]*t)
		return np.real(xh)
	return xh

# Example
f0 = 1
zeta = 0.2
omega0 = 2*np.pi*f0
a = np.array([omega0**2, 2*zeta*omega0, 1])
x0 = np.array([1, 0])
xh = lode(a, x0)

t = np.arange(0, 10, 1e-5)

p.plot(t, xh(t))
p.show()