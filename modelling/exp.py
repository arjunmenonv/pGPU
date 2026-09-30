import matplotlib.pyplot as plt 
import numpy as np 
import math

def exp_np(x):
    return np.exp(x)

def exp_taylor(x, n=8):
    out = np.ones_like(x, dtype=np.float64)
    term = np.ones_like(x, dtype=np.float64)
    for i in range(1, n):
        term = term * x / i
        out += term
    return out 

def get_valid_range(x, y_true, y_approx, rtol):
    """
    Finds the contiguous interval [x_beg, x_end] containing x=0
    where the relative error is within rtol.
    """
    rel_err = np.abs(y_approx - y_true) / np.abs(y_true)
    within_tol = rel_err <= rtol

    zero_idx = np.argmin(np.abs(x))
    if not within_tol[zero_idx]:
        return None, None

    # Step left towards negative values
    left = zero_idx
    while left > 0 and within_tol[left - 1]:
        left -= 1

    # Step right towards positive values
    right = zero_idx
    while right < len(x) - 1 and within_tol[right + 1]:
        right += 1

    return x[left], x[right]

if __name__ == '__main__':
    # Use a dense grid so x_beg and x_end are accurately resolved
    x = np.linspace(-25, 25, num=50000)
    y_np = exp_np(x)
    tol_low = []
    tol_high = []
    n_cand = [5, 8, 16, 20, 28, 64]
    rtol = 1e-3

    print(f"{'n':>4} | {'x_beg':>10} | {'x_end':>10} | {'Span':>10}")
    print("-" * 43)

    plt.figure(figsize=(10, 6))
    for cand in n_cand:
        y_taylor = exp_taylor(x, n=cand)
        x_beg, x_end = get_valid_range(x, y_np, y_taylor, rtol)
        tol_low.append(x_beg)
        tol_high.append(x_end)
        span = (x_end - x_beg) if (x_beg is not None and x_end is not None) else 0.0
        print(f"{cand:4d} | {x_beg:10.4f} | {x_end:10.4f} | {span:10.4f}")

        # Plot relative error curve for each n
        rel_err = np.abs(y_taylor - y_np) / np.abs(y_np)
        plt.plot(x, rel_err, label=f'n={cand} [{x_beg:.2f}, {x_end:.2f}]')

    plt.axhline(y=rtol, color='r', linestyle='--', label=f'rtol={rtol}')
    plt.yscale('log')
    plt.ylim(1e-12, 1e2)
    plt.xlim(-25, 25)
    plt.xlabel('x')
    plt.ylabel('Relative Error |exp_taylor - exp_np| / exp_np')
    plt.title(f'Taylor Approximation Relative Error vs exp(x) (rtol={rtol})')
    plt.grid(True, which='both', ls=':', alpha=0.6)
    plt.legend()
    plt.savefig('./relative_errors.png', bbox_inches='tight')
    plt.close()
    print("\nSaved error comparison figure to ./relative_errors.png")