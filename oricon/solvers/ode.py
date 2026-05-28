from scipy.integrate import solve_ivp


def solve_ode(fun, t_span, y0, t_eval=None, method="RK45", **kwargs):
    
    return solve_ivp(fun, t_span, y0, t_eval=t_eval, method=method, **kwargs)
