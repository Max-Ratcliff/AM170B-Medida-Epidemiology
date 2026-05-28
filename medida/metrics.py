import numpy as np

def coefficient_error(true_coeffs, estimated_coeffs):
    """L2 error between two coefficient matrices."""
    return np.linalg.norm(true_coeffs - estimated_coeffs)

def relative_error(true_traj, estimated_traj):
    """Mean relative L2 error over a trajectory."""
    diff = np.linalg.norm(true_traj - estimated_traj, axis=-1)
    norm = np.linalg.norm(true_traj, axis=-1)
    return np.mean(diff / (norm + 1e-12))

def format_equation(coeffs, feature_names, var_name="du/dt", tol=1e-4, precision=4):
    """Format a single ODE row as a string."""
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) > tol:
            sign = "+" if c > 0 else "-"
            val = abs(c)
            if val == 1.0:
                s_val = ""
            else:
                s_val = f"{val:.{precision}g}*"
            
            terms.append(f"{sign} {s_val}{feature_names[i]}")
    
    if not terms:
        return f"{var_name} = 0"
    
    res = " ".join(terms)
    if res.startswith("+ "):
        res = res[2:]
    elif res.startswith("- "):
        res = "-" + res[2:]
        
    return f"{var_name} = {res}"

def format_system(coeffs, feature_names, state_names=None, tol=1e-4, precision=4):
    """Format a full ODE system as a string."""
    n_vars = coeffs.shape[1]
    if state_names is None:
        state_names = [f"u{i}" for i in range(n_vars)]
    
    lines = []
    for j in range(n_vars):
        lines.append(format_equation(coeffs[:, j], feature_names, 
                                    var_name=f"d{state_names[j]}/dt", 
                                    tol=tol, precision=precision))
    return "\n".join(lines)
