import matplotlib.pyplot as plt
import numpy as np

# x values (right side, in mm)
x = np.array([1.6, 0.8, 0.4, 0.2, 0.1, 0.05, 0.025])

# y values (left side)
y = np.array([1.85, 1.80, 1.76, 1.73, 1.60, 1.41, 1.20])

#fit to \sqrt{D} to see if it matches the expected behavior
from scipy.optimize import curve_fit
def sqrt_func(D, a, b):
    return a * np.sqrt(D) + b
params, covariance = curve_fit(sqrt_func, x, y)
a_fit, b_fit = params
print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o')
# plt.plot(x, sqrt_func(x, *params), label=f'Fit: {a_fit:.2f}*sqrt(D) + {b_fit:.2f}', linestyle='--')

# Labels
plt.xlabel("D (mm)")
plt.ylabel("Y values")
plt.title("Y vs D")

# Grid
plt.grid(True)

# Optional: logarithmic x-axis since values halve each step
plt.xscale('log')
plt.yscale("log")

plt.show()