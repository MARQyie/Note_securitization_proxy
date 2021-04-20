#--------------------------------------------
# Make functions for semopy
#--------------------------------------------

# New functions
# MODEL CLASS
def calc_weight_wls(self):
    """
    Calculate the weight matrix of WLS/ADF

    Returns
    -------
    None
    """
    N = self.n_samples
    s = self.mx_cov
    s_tri = s[np.triu_indices(self.n_obs)]
    xtx_tri = N * s_tri
    
    # Calculate weight matrix
    Wijkl = np.einsum('i,j->ij', xtx_tri, xtx_tri) / N 
    Wij = np.einsum('i,j->ij', s_tri, s_tri)
    
    self.mx_w = Wijkl - Wij
    
    try:
        self.mx_w_inv  = np.linalg.inv(self.mx_w)
    except np.linalg.LinAlgError:
        self.mx_w_inv = np.inf
    
def obj_wls(self, x: np.ndarray):
    """
    Calculate WLS/ADF objective value.

    Parameters
    ----------
    x : np.ndarray
        Parameters vector.

    Returns
    -------
    float
        WLS value.
    """
    
    self.update_matrices(x)
    try:
        sigma, _ = self.calc_sigma()
    except np.linalg.LinAlgError:
        return np.inf
    
    if not hasattr(self, 'mx_w_inv'):
        self.calc_weight_wls()
    W_inv = self.mx_w_inv
         
    # Calculate the loss function
    svec = self.mx_cov[np.triu_indices(self.n_obs)]
    sigmavec = sigma[np.triu_indices(self.n_obs)]
    
    loss = (svec - sigmavec) @ W_inv @ (svec - sigmavec).T
    
    return loss

def grad_wls(self, x: np.ndarray):
    """
    Gradient of WLS/ADF objective function.

    Parameters
    ----------
    x : np.ndarray
        Parameters vector.

    Returns
    -------
    np.ndarray
        Gradient of WLS.

    """
    
    self.update_matrices(x)
    try:
        sigma, (m, c) = self.calc_sigma()
    except np.linalg.LinAlgError:
        t = np.zeros((len(x),))
        t[:] = np.inf
        return t
    sigma_grad = self.calc_sigma_grad(m, c)
    
    if not hasattr(self, 'mx_w_inv'):
        self.calc_weight_wls()
    W_inv = self.mx_w_inv
    
    svec = self.mx_cov[np.triu_indices(self.n_obs)]
    sigmavec = sigma[np.triu_indices(self.n_obs)]
    
    return 2 * np.array([(g[np.triu_indices(self.n_obs)] - sigmavec) @ W_inv @ (svec - sigmavec).T
                         for g in sigma_grad])

def __init__(self, description: str, mimic_lavaan=False, baseline=False):
        """
        Instantiate Model without mean-structure.

        Parameters
        ----------
        description : str
            Model description in semopy syntax.

        mimic_lavaan: bool
            If True, output variables are correlated and not conceptually
            identical to indicators. lavaan treats them that way, but it's
            less computationally effective. The default is False.

        baseline : bool
            If True, the model will be set to baseline model.
            Baseline model here is an independence model where all variables
            are considered to be independent with zero covariance. Only
            variances are estimated. The default is False.

        Returns
        -------
        None.

        """
        self.mimic_lavaan = mimic_lavaan
        self.parameters = dict()
        self.n_param_reg = 0
        self.n_param_cov = 0
        self.baseline = baseline
        self.constraints = list()
        dops = self.dict_operations
        dops[self.symb_starting_values] = self.operation_start
        dops[self.symb_bound_parameters] = self.operation_bound
        dops[self.symb_constraint] = self.operation_constraint
        self.objectives = {'MLW': (self.obj_mlw, self.grad_mlw),
                           'ULS': (self.obj_uls, self.grad_uls),
                           'GLS': (self.obj_gls, self.grad_gls),
                           'WLS': (self.obj_wls, self.grad_wls),
                           'FIML': (self.obj_fiml, self.grad_fiml)}
        super().__init__(description)