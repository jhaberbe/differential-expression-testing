import torch
import torch.nn
import torch.distributions

import pandas as pd

class BaseDistribution:

    def __init__(self):
        # Data placeholders
        self.X = None
        self.y = None

        # Names of design and feature variables
        self.design_names = []
        self.feature_names = []

        # Model parameters
        self.parameters = torch.nn.ParameterDict()
    
    @property
    def n_design(self) -> int:
        return len(self.design_names)
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    def setup(self, X: pd.DataFrame, y: pd.DataFrame):
        self.design_names = X.columns.tolist()
        self.feature_names = y.columns.tolist()
        self._initialize_parameters(X, y)
        self.X = torch.tensor(X.values)
        self.y = torch.tensor(y.values)
    
    def _initialize_parameters(self, X: pd.DataFrame, y: pd.DataFrame):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def log_likelihood(self, X: pd.DataFrame, y: pd.DataFrame) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def export_parameters(self) -> torch.nn.ParameterDict:
        raise NotImplementedError("Subclasses should implement this method.")


class NegativeBinomial(BaseDistribution):
    # ... your existing __init__ / setup / etc. ...

    def _initialize_parameters(self, X: pd.DataFrame, y: pd.DataFrame):
        # assumes you use beta with shape (n_design, n_features)
        self.parameters['beta'] = torch.nn.Parameter(
            torch.zeros(self.n_design, self.n_features)
        )
        self.parameters['log_dispersion'] = torch.nn.Parameter(
            torch.zeros(self.n_features)
        )

    def log_likelihood(self, X: pd.DataFrame, y: pd.DataFrame) -> torch.Tensor:
        beta = self.parameters['beta']                          # (p, f)
        log_r = self.parameters['log_dispersion']               # (f,)
        log_mu = self.X @ beta                                  # (n, f)
        logits = log_r - log_mu                                 # (n, f); p/(1-p)=r/μ
        r = torch.exp(log_r)
        dist = torch.distributions.NegativeBinomial(total_count=r, logits=logits)
        return dist.log_prob(self.y).sum()

    def _beta_variance_diag(self, damping: float = 1e-3) -> torch.Tensor:
        """
        Empirical Fisher diagonal for beta:
            F_diag = sum_i (∂NLL_i/∂beta)^2
        Var(beta) ≈ 1 / (F_diag + damping), shape (n_design, n_features).
        """
        # ensure grad is enabled even if caller is under no_grad
        with torch.enable_grad():
            beta = self.parameters['beta']                     # requires_grad=True
            fim_diag = torch.zeros_like(beta)

            n = self.X.shape[0]
            for i in range(n):
                Xi = self.X[i:i+1, :]                          # (1, p)
                yi = self.y[i:i+1, :]                          # (1, f)

                log_r = self.parameters['log_dispersion']      # (f,)
                log_mu_i = Xi @ beta                           # (1, f)
                logits_i = log_r - log_mu_i                    # (1, f)
                r = torch.exp(log_r)
                dist_i = torch.distributions.NegativeBinomial(total_count=r, logits=logits_i)
                nll_i = -dist_i.log_prob(yi).sum()

                (g_beta,) = torch.autograd.grad(nll_i, (beta,), retain_graph=False, create_graph=False)
                fim_diag += g_beta**2

            var_beta = 1.0 / (fim_diag + damping)
            return var_beta

    def export_parameters(self) -> pd.DataFrame:
        # compute variance first (needs grad)
        beta_var = self._beta_variance_diag().detach().cpu().numpy()   # (p, f)

        # then safely detach values for the table
        with torch.no_grad():
            beta = self.parameters['beta'].detach().cpu().numpy()       # (p, f)

        # transpose so rows = features, cols = designs
        beta_T = beta.T        # (f, p)
        beta_var_T = beta_var.T

        beta_df = pd.DataFrame(beta_T, index=self.feature_names, columns=self.design_names)
        beta_var_cols = [f"{d} variance" for d in self.design_names]
        beta_var_df = pd.DataFrame(beta_var_T, index=self.feature_names, columns=beta_var_cols)

        return pd.concat([beta_df, beta_var_df], axis=1)