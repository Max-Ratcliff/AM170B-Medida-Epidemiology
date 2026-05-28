import numpy as np

def coefficient_error(true_coeffs, estimated_coeffs):
    """Normalised coefficient distance || c_s - c_other || / || c_s || (Eq. 16)."""
    true_coeffs = np.asarray(true_coeffs, dtype=float)
    estimated_coeffs = np.asarray(estimated_coeffs, dtype=float)
    denom = np.linalg.norm(true_coeffs)
    if denom == 0.0:
        return float(np.linalg.norm(true_coeffs - estimated_coeffs))
    return float(np.linalg.norm(true_coeffs - estimated_coeffs) / denom)

def relative_error(reference, estimate):
    """Generic relative L2 error || reference - estimate || / || reference ||."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return float(np.linalg.norm(reference - estimate))
    return float(np.linalg.norm(reference - estimate) / denom)

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
