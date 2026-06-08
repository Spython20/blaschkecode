import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


# step 1
def make_fingerprint(roots, rot=0.0, Nfp=4000):
    roots = np.asarray(roots, complex)
    n = len(roots) # deg B

    def B(t): # Blaschke product
        z = np.exp(1j * np.asarray(t, float)) # z = exp[it_j], creates array of angles t into points on unit circle
        out = np.full(z.shape, np.exp(1j * rot), complex) # creates array with rotation e^{i rot} in each position
        for a in roots:
            out = out * (z - a) / (1 - np.conjugate(a) * z) # multiplies every matrix entry by blaschke roots, giving array of blascke products evaluated at each sample point t
        return out

    t_full = np.linspace(0, 2 * np.pi, Nfp + 1)
    alpha = np.unwrap(np.angle(B(t_full))) # np.angle only gives values from (-pi, pi] so unwrap makes alpha continuous
    psi_vals = alpha / n 

    _psi = PchipInterpolator(t_full, psi_vals) # interpolates to determine psi, not sure if valid or better way
    _chi = PchipInterpolator(psi_vals, t_full) # inverse of psi
    psi0 = psi_vals[0]

    def psi(x):
        x = np.asarray(x, float) # turns input into array
        m = np.floor(x / (2 * np.pi))
        return _psi(x - 2 * np.pi * m) + 2 * np.pi * m

    def chi(p):
        p = np.asarray(p, float)
        m = np.floor((p - psi0) / (2 * np.pi))
        return _chi(p - 2 * np.pi * m) + 2 * np.pi * m

    return B, psi, chi

# step 2

# what i use to interpolate F, could be optimized, approximates F(eval_pts), got from AI
def trig_interp_matrix(eval_pts, M):
    xj = 2 * np.pi * np.arange(M) / M
    X = eval_pts[:, None] - xj[None, :]
    Xr = (X + np.pi) % (2 * np.pi) - np.pi

    small = np.abs(Xr) < 1e-12

    with np.errstate(divide='ignore', invalid='ignore'):
        S = (1.0 / M) * np.sin(M * Xr / 2) / np.tan(Xr / 2)

    S[small] = 1.0
    return S

# useful function to wrap angles to (-pi, pi], got from AI after getting bad results
def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def weld(psi, chi, M=512):
    theta = 2 * np.pi * np.arange(M) / M # creates grid points ie angles we sample to find F
    h = 2 * np.pi / M # grid spacing

    # boundaries of interval I defined in Mumford
    theta_minus = theta - h / 2 
    theta_plus = theta + h / 2

    chi_theta = chi(theta)
    chi_minus = chi(theta_minus)
    chi_plus = chi(theta_plus)

    A = theta[:, None] # makes theta a column matrix for numpy

    # makes theta minus and theta plus row vectors for numpy
    Bm = theta_minus[None, :]
    Bp = theta_plus[None, :]

    CA = chi_theta[:, None] # makes chi theta a column matrix for numpy

    # makes chi minus and chi plus row vectors for numpy
    Cm = chi_minus[None, :]
    Cp = chi_plus[None, :]

    # The code below (lines 84-94) computes the individual "components" of the integrated cotangent-kernel K matrix from Sharon-Mumford using the normalized Hilbert transform convention.

    # These are wrapped so that sin is evaluated faster, got it from chatgpt after my code was taking forever and getting weird errors
    s_theta_m = np.sin(wrap_angle(A - Bm) / 2)
    s_theta_p = np.sin(wrap_angle(A - Bp) / 2)

    s_chi_m = np.sin(wrap_angle(CA - Cm) / 2)
    s_chi_p = np.sin(wrap_angle(CA - Cp) / 2)

    ratio = np.abs((s_theta_m * s_chi_p) / (s_theta_p * s_chi_m))

    tiny = np.finfo(float).tiny
    ratio = np.maximum(ratio, tiny) # if ever get log(0) from numerical inconsistencies, evaluates at nonzero extremely small value instead

    K = (1j / (2 * np.pi)) * np.log(ratio) # i/2pi term comes from Hilbert normallization convention H(e^intheta) = -i sign(n) e^(in theta)

    F = np.linalg.solve(np.eye(M, dtype=complex) + K, np.exp(1j * theta))

    P = trig_interp_matrix(psi(theta), M)    # f_-(e^{i theta}) = f_+(e^{i psi(theta)}), so f_-(e^itheta) = F(\psi(theta)), thus PF approxeq f_-(e^itheta)
  # the line above is NOT the final polynomial, it is the interpolation matrix
    return theta, F, P


# step 3
def recover_polynomial(theta, F, P_mat, B, degree):
    f_minus_trace = P_mat @ F # samples f_-(e^{i theta_j}) = F(psi(theta_j))

    A = np.vstack([f_minus_trace ** k for k in range(degree + 1)]).T # builds least squares matrix
    
    coef, *_ = np.linalg.lstsq(A, B(theta), rcond=None) # solves the system of equations from A, giving us the coefficients of the final polynomial
    
    Ppoly = lambda w: sum(coef[k] * np.asarray(w, complex) ** k for k in range(degree + 1)) # creates the final polynomial
    
    return coef, Ppoly

# Below, we run all of the code finally

# roots of blaschke product
roots = [
    0.6j,
    0.3,
    0.8j,
    0.9,
    0.9j
]

B, psi, chi = make_fingerprint(roots, rot=0.0, Nfp=60000)
theta, F, P_mat = weld(psi, chi, M=2048)
coef, Ppoly = recover_polynomial(theta, F, P_mat, B, degree=len(roots))

# print the polynomial
print("P coefficients [c0..cn]       :", np.round(coef, 5))
print("P roots                       :", np.round(np.roots(coef[::-1]), 5))

# plot generation
g = np.linspace(F.real.min() - .3, F.real.max() + .3, 700)
X, Y = np.meshgrid(g, g)
Z = np.abs(Ppoly(X + 1j * Y))

fig, ax = plt.subplots(figsize=(6, 6))

ax.contourf(X, Y, Z, levels=[0, 1], colors=["#cfe8ff"], alpha=.7)
ax.contour(X, Y, Z, levels=[1], colors=["#9abfe0"], linewidths=1.0)

ax.plot(np.append(F.real, F.real[0]), np.append(F.imag, F.imag[0]),
        color="#1f4e9c", lw=1.8, ls="--", label="welded curve")

# Unit circle
u = np.linspace(0, 2*np.pi, 1000)
ax.plot(np.cos(u), np.sin(u),
        color="green", lw=1.8, ls="--", label="unit circle")

# roots of P
rt = np.roots(coef[::-1])
ax.scatter(rt.real, rt.imag, c="#c0392b", s=55, zorder=5, label="roots of P")

# Blaschke roots
br = np.asarray(roots, complex)
ax.scatter(br.real, br.imag,
           c="purple", marker="o", s=55, linewidths=2,
           zorder=6, label="Blaschke roots")

ax.set_aspect("equal")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.set_title("Lemniscate (Jordan Curve) $\\Gamma = \\{|P|=1\\}$")
plt.tight_layout()
plt.show()
