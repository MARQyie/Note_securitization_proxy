#--------------------------------------------
# Make functions for semopy
#--------------------------------------------
# Note, these function allow a custom w during WLS
def fit(self, data=None, cov=None, obj='MLW', solver='SLSQP', groups=None,
            clean_slate=False, regularization=None, n_samples=None, custom_w = None **kwargs):
        """
        Fit model to data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data with columns as variables. The default is None.
        cov : pd.DataFrame, optional
            Pre-computed covariance/correlation matrix. The default is None.
        obj : str, optional
            Objective function to minimize. Possible values are 'MLW', 'FIML',
            'ULS', 'GLS', 'WLS', 'DWLS'. The default is 'MLW'.
        solver : str, optional
            Optimizaiton method. Currently scipy-only methods are available.
            The default is 'SLSQP'.
        groups : list, optional
            Groups of size > 1 to center across. The default is None.
        clean_slate : bool, optional
            If False, successive fits will be performed with previous results
            as starting values. If True, parameter vector is reset each time
            prior to optimization. The default is False.
        regularization
            Special structure as returend by create_regularization function.
            If not None, then a regularization will be applied to a certain
            parameters in the model. The default is None.
        n_samples : int, optional
            Number of samples in data. Used only if data is None and cov is
            provided for Fisher Information Matrix calculation. The default is
            None.

        Raises
        ------
        Exception
            Rises when attempting to use FIML in absence of full data.

        Returns
        -------
        SolverResult
            Information on optimization process.

        """
        self.load(data=data, cov=cov, groups=groups,
                  clean_slate=clean_slate, n_samples=n_samples)
        if obj == 'FIML':
            if not hasattr(self, 'mx_data'):
                raise Exception('Full data must be supplied for FIML')
            self.prepare_fiml()
        elif obj in ('WLS', 'DWLS'):
            if (not hasattr(self, 'last_result')) or \
                (self.last_result.name_obj != obj):
                    self.prepare_wls(obj, custom_w)
        fun, grad = self.get_objective(obj, regularization=regularization)
        solver = Solver(solver, fun, grad, self.param_vals,
                        constrs=self.constraints,
                        bounds=self.get_bounds(), **kwargs)
        res = solver.solve()
        res.name_obj = obj
        self.param_vals = res.x
        self.update_matrices(res.x)
        self.last_result = res
        return res

def prepare_wls(self, obj: str, custom_w=None):
        """
        Prepare data structures for efficient WLS/DWLS estimation.

        Parameters
        ----------
        obj : str
            Either 'WLS' or 'DWLS'.
        custom_w : np.ndarray, optional
            Optional custom weight matrix. The default is None.

        Returns
        -------
        None.

        """
        
		if custom_w is not None:
			w = custom_w
		else:
			data = self.mx_data - self.mx_data.mean(axis=0)
			products = list()
			for i in range(data.shape[1]):
				for j in range(i, data.shape[1]):
					products.append(data[:, i] * data[:, j])
			products = np.array(products)
			w = np.cov(products, bias=True)
		
		if obj == 'DWLS':
			self.mx_w_inv = np.array([1 / d for d in w.diagonal()])
		else:
			try:
				self.mx_w_inv = np.linalg.inv(w)
			except np.linalg.LinAlgError:
				logging.warning("Weight matrix could not be inverted. NearPD"
								" estimate will be used instead.")
				w = cov_nearest(w, threshold=1e-2)
				self.mx_w_inv = np.linalg.pinv(w)
		self.mx_w = w
		self.inds_triu_sigma = np.triu_indices_from(self.mx_cov)
		self.mx_vech_s = self.mx_cov[self.inds_triu_sigma]

#--------------------------------------------
# OLD
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