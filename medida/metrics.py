import numpy as np


def coefficient_error(true_coeffs, estimated_coeffs):
    """Normalised coefficient distance || c_s - c_other || / || c_s || (Eq. 16).

    Args:
        true_coeffs (array-like): The ground truth coefficient matrix.
        estimated_coeffs (array-like): The discovered/estimated coefficients.

    Returns:
        float: The normalised error (0.0 means perfect recovery).
    """
    true_coeffs = np.asarray(true_coeffs, dtype=float)
    estimated_coeffs = np.asarray(estimated_coeffs, dtype=float)
    denom = np.linalg.norm(true_coeffs)
    if denom == 0.0:
        return float(np.linalg.norm(true_coeffs - estimated_coeffs))
    return float(np.linalg.norm(true_coeffs - estimated_coeffs) / denom)


def relative_error(reference, estimate):
    """Generic relative L2 error || reference - estimate || / || reference ||.

    Args:
        reference (array-like): The baseline/true values.
        estimate (array-like): The estimated/predicted values.

    Returns:
        float: The relative error.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return float(np.linalg.norm(reference - estimate))
    return float(np.linalg.norm(reference - estimate) / denom)


def format_equation(
    coeffs, feature_names, var_name="du/dt", tol=1e-4, precision=4
):
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


def format_system(
    coeffs, feature_names, state_names=None, tol=1e-4, precision=4
):
    """Format a full ODE system as a string."""
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
        var_name = state_names[0] if state_names else "u"
        return format_equation(
            coeffs,
            feature_names,
            var_name=f"d{var_name}/dt",
            tol=tol,
            precision=precision,
        )

    n_features, n_vars = coeffs.shape
    if state_names is None:
        state_names = [f"u{i}" for i in range(n_vars)]

    lines = []
    for j in range(n_vars):
        lines.append(
            format_equation(
                coeffs[:, j],
                feature_names,
                var_name=f"d{state_names[j]}/dt",
                tol=tol,
                precision=precision,
            )
        )
    return "\n".join(lines)


def to_latex(name):
    """Convert a feature name to LaTeX (e.g., u^2 -> u^{2}, u_xx -> u_{xx})."""
    if name == "1":
        return "1"
    name = name.replace("^", "^{").replace(" ", " ")
    if "^{" in name:
        # Closing the bracket for exponents - simple heuristic
        parts = name.split(" ")
        new_parts = []
        for p in parts:
            if "^{" in p:
                new_parts.append(p + "}")
            else:
                new_parts.append(p)
        name = " ".join(new_parts)

    if "_" in name:
        name = name.replace("_", "_{") + "}"
    return name


def format_latex_system(
    coeffs, feature_names, state_names=None, tol=1e-4, precision=4
):
    """Generate raw LaTeX text for the discovered system."""
    coeffs = np.asarray(coeffs, dtype=float)
    if state_names is None:
        if coeffs.ndim == 1:
            n_vars = 1
        else:
            n_vars = coeffs.shape[1]
        state_names = [f"u_{i}" for i in range(n_vars)]

    def _format_row(row_coeffs, var_name):
        terms = []
        for i, c in enumerate(row_coeffs):
            if abs(c) > tol:
                sign = "+" if c > 0 else "-"
                val = abs(c)
                s_val = f"{val:.{precision}g}" if val != 1.0 else ""
                feat = to_latex(feature_names[i])
                terms.append(f"{sign} {s_val} {feat}")

        if not terms:
            return f"\\frac{{d{var_name}}}{{dt}} &= 0"
        res = " ".join(terms)
        if res.startswith("+ "):
            res = res[2:]
        elif res.startswith("- "):
            res = "-" + res[2:]
        return f"\\frac{{d{var_name}}}{{dt}} &= {res}"

    lines = []
    if coeffs.ndim == 1:
        lines.append(_format_row(coeffs, state_names[0]))
    else:
        for j in range(coeffs.shape[1]):
            lines.append(_format_row(coeffs[:, j], state_names[j]))

    out = "\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}"
    return out
