#load packages
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import matplotlib.pyplot as plt
import seaborn as sns
import adjustText
from sksurv.metrics import concordance_index_censored
from sklearn.utils import resample
import joblib
from joblib import Parallel, delayed
from sksurv.metrics import cumulative_dynamic_auc

class ReCAST:
    def __init__(self, n_models = 100, 
                 l1 = 1, 
                 val_size = 0.3, 
                 n_folds = 3, 
                 bootstrap = True, 
                 random_state = None, 
                 adaptive_lasso = True, 
                 normalization = None, 
                 metric = 'c-index', 
                 auc_time_range=None, 
                 n_jobs = -2, 
                 subagging = 1, 
                 verbose = True,
                 ):
        self.n_models = n_models
        self.l1 = l1
        self.val_size = val_size
        self.n_folds = n_folds
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.models_ = []
        self.risk_scores_train_matrix = []
        self.risk_scores_train = []
        self.risk_scores_train_non_normalized = []
        self.coefs_ = []
        self.coef_mean_ = []
        self.fitted_ = False
        self.train_risk_min_ = None
        self.train_risk_max_ = None
        self.adaptive_lasso = adaptive_lasso
        self.metric = metric
        self.auc_time_range = auc_time_range
        self.normalization = normalization
        self.subagging = subagging
        self.n_jobs = n_jobs
        self.verbose = verbose
    '''
    n_models: number of base learners included in the framework. Deafult is n=100
    l1: float between 0 and 1, the elastic net mixing parameter. l1=1 corresponds to Lasso penalty, l1=0 to Ridge penalty. Default is l=1
    val_size: float between 0 and 1, proportion of data to use as validation set if bootstrap=False
    n_folds: number of folds for cross-validation for hyperparameter tuning within each model
    bootstrap: whether to use bootstrap sampling (True) or train/validation split (False) for fitting individual models
    random_state: random seed for reproducibility
    adaptive_lasso: whether to use adaptive lasso penalty with data-driven penalty factors
    normalization: None or tuple of two floats between 0 and 1 to specify quantiles for min-max normalization of risk scores. If None, use min and max of training risk scores.
    metric: 'c-index' or 'auc' to specify metric for hyperparameter tuning. If 'auc', use time-dependent AUC with times defined by auc_time_range or default percentiles of event times.
    auc_time_range: tuple of two floats to specify lower and upper time for calculating time-dependent AUC during hyperparameter tuning. If None, use 10th and 80th percentiles of event times in training data.
    n_jobs: number of parallel jobs to run for fitting models. -1 means using all processors, -2 means using all but one processor.
    subagging: float between 0 and 1, proportion of samples to use for each bootstrap sample when bootstrap=True. Default is 1 (full bootstrap).
    verbose: whether to print progress messages during fitting
    '''

    
    def _fit_single_model(self, i, X, y):
        """
        Helper method to fit a single model. This is what runs in parallel.
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        current_random_state = None if self.random_state is None else self.random_state + i

        #### BOOTSTRAP VS TRAIN/VAL SPLIT
        if self.bootstrap:
            X_tr, y_tr = resample(X, y, replace=True, n_samples=int(len(X) * self.subagging), random_state=current_random_state)
            # out-of-bag samples for validation
            oob_mask = ~X.index.isin(X_tr.index)
            X_val = X.loc[oob_mask]
            y_val = y.loc[oob_mask]
            
            # If no OOB samples, skip this iteration (return None)
            if X_val.shape[0] == 0:
                return None
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=self.val_size, stratify=y['Event'], random_state=current_random_state
            )

        # find alphas
        y_tr_surv = Surv.from_dataframe('Event', 'Time', y_tr)
        
        ### SCORING METHOD, C-index vs Time dependent AUC
        scoring_method = None
        
        if self.metric == 'auc':
            # time points
            event_times = y_tr['Time'][y_tr['Event'] == 1]
            if len(event_times) < 5: return None 
            
            if self.auc_time_range: 
                lower_t = self.auc_time_range[0] 
                upper_t = self.auc_time_range[1]
            else:
                lower_t = np.percentile(event_times, 0.1 * 100)
                upper_t = np.percentile(event_times, 0.8 * 100)
            time_diff = upper_t - lower_t
            if time_diff <= 24:
                time_bins = 5
            else:
                time_bins = 10
            times = np.linspace(lower_t, upper_t, time_bins)
            
            def cumulative_auc_wrapper(y_true, y_pred, **kwargs):
                auc, mean_auc = cumulative_dynamic_auc(y_tr_surv, y_true, y_pred, times)
                return mean_auc

            scoring_method = make_scorer(cumulative_auc_wrapper, greater_is_better=True, needs_threshold=False)
            
        
        ### ADAPTIVE LASSO PENALTY FACTORS
        penalty_factors = None 
        
        if self.adaptive_lasso:
            ridge_cv = GridSearchCV(
                CoxnetSurvivalAnalysis(l1_ratio=0.001), 
                param_grid={'alphas': [[v] for v in np.logspace(-2, 1, 100)]},
                cv=3,
                n_jobs=1,
                error_score='raise', 
                scoring=scoring_method
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    ridge_cv.fit(X_tr, y_tr_surv)
                except Exception as e:
                    print(f"Iter {i}: Ridge fit failed: {e}")
                    return None
  
            best_ridge = ridge_cv.best_estimator_
            coefs_initial = best_ridge.coef_.squeeze()

            abs_coefs = np.abs(coefs_initial)            
            clip_floor = abs_coefs.min() + 1e-3
            clip_ceil = abs_coefs.max() 
            abs_coefs_clipped = np.clip(abs_coefs, a_min=clip_floor, a_max=clip_ceil)

            epsilon = 1e-4
            weights = 1.0 / (abs_coefs_clipped + epsilon)

            penalty_factors = weights / weights.mean()
            

        # find alphas
        alpha_finder = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1, 
            alpha_min_ratio=0.01, 
            n_alphas=100, 
            penalty_factor=penalty_factors
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                alpha_finder.fit(X_tr, y_tr_surv)
            except Exception as e:
                # print(f"Error fitting alpha finder: {e}")
                return None

        gs = GridSearchCV(
            CoxnetSurvivalAnalysis(
                l1_ratio=self.l1, 
                fit_baseline_model=True, 
                penalty_factor=penalty_factors
            ),
            param_grid={'alphas': [[a] for a in alpha_finder.alphas_]},
            cv=self.n_folds,
            n_jobs=1,
            scoring=scoring_method,
            error_score='raise'
        )
        
        ### FIT

        #fallback function for convergence warnings during fitting, and in case input manual alphas
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                #automatically found alphas
                gs.fit(X_tr, y_tr_surv)
            except Exception as e:
                if self.verbose:
                    print(f"Iter {i}: Auto-alpha failed ({e}). Retrying with manual alphas...")
                
                try:
                    # manual range                
                    manual_alphas = [[v] for v in np.logspace(-1, 2, 100)]
                    gs = GridSearchCV(
                        CoxnetSurvivalAnalysis(
                            l1_ratio=self.l1, 
                            fit_baseline_model=True, 
                            penalty_factor=penalty_factors
                        ),
                        param_grid={'alphas': manual_alphas},
                        cv=self.n_folds,
                        n_jobs=1,
                        scoring=scoring_method,
                        error_score='raise'
                    )
                    gs.fit(X_tr, y_tr_surv)
                    
                except Exception as e2:
                    if self.verbose:
                        print(f"Iter {i}: Manual-alpha also failed ({e2}). Skipping this model.")
                    return None
            

        best_model = gs.best_estimator_    
        # get coeffs
        this_coef = pd.Series(best_model.coef_.squeeze(), index=X.columns, name=f'model {i+1}')
        
        # predict on validation set
        risk_val = best_model.predict(X_val)
        this_risk_score = pd.DataFrame({
            'index': X_val.index, 
            f'risk_score model {i+1}': risk_val
        }).set_index('index')

        #get single models c-index
        c_index_model = concordance_index_censored(y_val['Event'].astype(bool), y_val['Time'], risk_val)[0]

        return best_model, this_coef, this_risk_score, c_index_model

    def fit(self, X, y):
        if self.verbose:
            print(f"Fitting {self.n_models} CoxNet Models...")        
        if 'Event' not in y.columns or 'Time' not in y.columns:
            raise ValueError("y must have columns 'Event' and 'Time'")
        if y.isnull().values.any():
            raise ValueError("y contains missing values. Remove or impute them first.")
        
        if self.val_size >=0.5 and not self.bootstrap:
            if self.verbose:
                print(f'### Warning: val_size is above half the dataset which may lead to small training sets and failure of models to converge ###')

        if self.bootstrap:
            if self.verbose:
                print(f'Selected `bootstrap` sampling for model fitting')
        else:
            if self.verbose:
                print(f'Selected train/validation split for model fitting')
        
        if self.adaptive_lasso:
            if self.verbose:
                print('Selected the option `adaptive_lasso` to assign distinct penalty to different coefficients')
        
        if self.metric == 'auc':
            if self.verbose:
                print(f'Using time-dependent AUC as model selection metric...')
            if self.auc_time_range:
                if self.verbose:
                    print(f'Using user-defined AUC time range for hyperparameter tuning: {self.auc_time_range}')
            else:
                if self.verbose:
                    print('Using default AUC time range: 10th to 80th percentile of event times for hyperparameter tuning')
        else:
            if self.verbose:
                print(f'Using default Harrel c-index for hyperparameter tuning. If AUC desired, use `metric="auc"`')

        # --- PARALLEL EXECUTION ---
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(i, X, y) 
            for i in range(self.n_models)
        )
        results = [r for r in results if r is not None]
        
        if not results:
            raise RuntimeError("All models failed to fit. Check your data or parameters.")

        #unpack results
        self.models_, coefs_list, risk_scores_list, c_index_list = zip(*results)
        self.models_ = list(self.models_) 
        coefs = pd.concat(coefs_list, axis=1)
        self.risk_scores_train_matrix = pd.concat(risk_scores_list, axis=1)
        self.risk_scores_train_non_normalized = self.risk_scores_train_matrix.mean(axis=1, skipna=True).rename('risk_score_non_normalized')
        self.cindex_models = pd.Series(c_index_list, index=[f'model {i+1}' for i in range(len(c_index_list))], name='c_index')
        
        # average Risk Scores
        self.risk_scores_train = self.risk_scores_train_matrix.mean(axis=1, skipna=True).rename('risk_score')
        
        self.coefs_ = coefs 
        self.coef_mean_ = self.coefs_.mean(axis=1)

        #normalization using selected quantiles if selected
        if self.normalization is not None:
            self.train_risk_min_ = float(self.risk_scores_train.quantile(self.normalization[0]))
            self.train_risk_max_ = float(self.risk_scores_train.quantile(self.normalization[1]))
            denom = (self.train_risk_max_ - self.train_risk_min_)

        # normalization using min and max if no quantiles provided
        if self.normalization is None:
            self.train_risk_min_ = float(self.risk_scores_train.min())
            self.train_risk_max_ = float(self.risk_scores_train.max())
            denom = (self.train_risk_max_ - self.train_risk_min_)

        if denom == 0:
            self.risk_scores_train = self.risk_scores_train * 0 + 0
        else:
            self.risk_scores_train = ((self.risk_scores_train - self.train_risk_min_) / denom * 100.0).clip(0, 100)

        self.fitted_ = True
        if self.verbose:
            print(f"Finished fitting {len(self.models_)} models.")



    def define_3_risk_groups(self, plot = False, save = False, figsize=(6,4), bins=30, legend = False, detailed = False):
        '''
        Use Gaussian Mixture Model to define 3 risk groups based on risk scores
        args: 
        plot: whether to plot the GMM fit and cutoffs
        save: whether to save the plot, if True provide path as string
        figsize: tuple for figure size if plot is True
        bins: number of bins for histogram if plot is True
        legend: whether to show legend in plot if plot is True
        detailed: whether to show detailed plot with individual GMM components and histogram (True) or a simpler SCORPIO-style plot with just the overall density and colored risk areas (False)
        '''
        if not self.fitted_:
            raise RuntimeError("You must fit the model before defining risk groups.")
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3, random_state=94)
        risk_scores_array = self.risk_scores_train.values.reshape(-1, 1)
        gmm.fit(risk_scores_array)
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        covariances = gmm.covariances_.flatten()[sorted_indices]
        weights = gmm.weights_[sorted_indices]

        def gmm_pdf(x, component_index):
            """Calculates the weighted probability density"""
            mu = means[component_index]
            sigma = np.sqrt(covariances[component_index])
            weight = weights[component_index]
            exponent = -0.5 * ((x - mu) / sigma) ** 2
            return weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
        def find_crossover(comp_a, comp_b, search_range):
            """Finds the point where the PDF of component A equals the PDF of component B."""
            from scipy.optimize import brentq
            def difference_function(x):
                return gmm_pdf(x, comp_a) - gmm_pdf(x, comp_b)
            try:
                return brentq(difference_function, search_range[0], search_range[1])
            except ValueError:
                return np.nan # Return NaN if root not found 
        # Crossover 1: Low (0) vs Intermediate (1)
        search_range_c1 = (means[0], means[1])
        cutoff_low_intermediate = find_crossover(0, 1, search_range_c1)
        # Crossover 2: Intermediate (1) vs High (2)
        search_range_c2 = (means[1], means[2])
        cutoff_intermediate_high = find_crossover(1, 2, search_range_c2)
        if plot:
            if detailed:
                #plotting function detailed
                x_min, x_max = np.min(risk_scores_array), np.max(risk_scores_array)
                x_range = np.linspace(x_min, x_max, 500)
                plt.figure(figsize=figsize)
                plt.hist(risk_scores_array, bins=bins, density=True, alpha=0.6, color='skyblue', label='Risk Score Histogram')
                gmm_total_density = np.zeros_like(x_range)
                for i in range(3):
                    pdf = gmm_pdf(x_range, i)
                    gmm_total_density += pdf
                    plt.plot(x_range, pdf, linestyle='--', label=f'Component {i+1} PDF (Mean: {means[i]:.0f})')
                plt.plot(x_range, gmm_total_density, color='blue', linewidth=1, label='Total GMM Density')
                if not np.isnan(cutoff_low_intermediate):
                    plt.axvline(cutoff_low_intermediate, color='green', linestyle='-', label=f'Cutoff C1 ({cutoff_low_intermediate:.2f})')
                if not np.isnan(cutoff_intermediate_high):
                    plt.axvline(cutoff_intermediate_high, color='#99000d', linestyle='-', label=f'Cutoff C2 ({cutoff_intermediate_high:.2f})')
                plt.title('GMM Fit')
                plt.xlabel('Risk Score')
                plt.ylabel('Density')
                if legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                if save:
                    import matplotlib as mpl
                    mpl.rcParams['pdf.fonttype'] = 42
                    plt.savefig(save, bbox_inches='tight')
                plt.show()
             ### SCORPIO STYLE
            else:
                plt.figure(figsize=figsize)
                x_min, x_max = np.min(risk_scores_array) - 0.05, np.max(risk_scores_array) + 0.05
                x_range = np.linspace(x_min, x_max, 500)
                gmm_total_density = np.zeros_like(x_range)
                for nr, i in enumerate(range(3)):
                    color = ["#127812", "#A87D11", "#951111"][nr]
                    pdf = gmm_pdf(x_range, i)
                    gmm_total_density += gmm_pdf(x_range, i)
                    plt.plot(x_range, pdf, linestyle='--', color=color, label=f'Component {i+1} PDF (Mean: {means[i]:.0f})')
                
                if legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)

                mask_low = x_range <= cutoff_low_intermediate
                mask_intermediate = (x_range > cutoff_low_intermediate) & (x_range <= cutoff_intermediate_high)
                mask_high = x_range > cutoff_intermediate_high

                color_low = '#228B22'   
                color_intermediate = '#EEB422'   
                color_high = '#FF3030'  

                plt.fill_between(x_range[mask_low], gmm_total_density[mask_low], color=color_low, alpha=0.5)
                plt.fill_between(x_range[mask_intermediate], gmm_total_density[mask_intermediate], color=color_intermediate, alpha=0.5)
                plt.fill_between(x_range[mask_high], gmm_total_density[mask_high], color=color_high, alpha=0.5)

                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.xlim(x_min, x_max) 
                plt.xlabel('Risk Score')
                plt.ylabel('Density')
                plt.margins(y=0)

                if save:
                    import matplotlib as mpl
                    mpl.rcParams['pdf.fonttype'] = 42
                    plt.savefig(save, bbox_inches='tight')
                
                plt.show()
        self.cutoff_low_intermediate_ = cutoff_low_intermediate
        self.cutoff_intermediate_high_ = cutoff_intermediate_high
        return print(f'cutoff between low and intermediate risk calculated at: {cutoff_low_intermediate:.2f}, cutoff between intermediate and high risk calculated at: {cutoff_intermediate_high:.2f}')
    
    def define_2_risk_groups(self, plot = False, save = False, figsize=(6,4), bins=30, legend = False, detailed = False):
        '''
        Use Gaussian Mixture Model to define 2 risk groups based on risk scores
        args:
        plot: whether to plot the GMM fit and cutoff
        save: whether to save the plot, if True provide path as string
        figsize: tuple for figure size if plot is True
        bins: number of bins for histogram if plot is True
        legend: whether to show legend in plot if plot is True
        detailed: whether to show detailed plot with individual GMM components and histogram (True) or a simpler SCORPIO-style plot with just the overall density and colored risk areas (False)
        '''
        from sklearn.mixture import GaussianMixture
        if not self.fitted_:
            raise RuntimeError("You must fit the model before defining risk groups.")
        gmm = GaussianMixture(n_components=2, random_state=94)
        risk_scores_array = self.risk_scores_train.values.reshape(-1, 1)
        gmm.fit(risk_scores_array)
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        covariances = gmm.covariances_.flatten()[sorted_indices]
        weights = gmm.weights_[sorted_indices] 
        def gmm_pdf(x, component_index):
            """Calculates the weighted probability density"""
            mu = means[component_index]
            sigma = np.sqrt(covariances[component_index])
            weight = weights[component_index]
            exponent = -0.5 * ((x - mu) / sigma) ** 2
            return weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(exponent)
        def find_crossover(comp_a, comp_b, search_range):
            """Finds the point where the PDF of component A equals the PDF of component B."""
            from scipy.optimize import brentq
            def difference_function(x):
                return gmm_pdf(x, comp_a) - gmm_pdf(x, comp_b)
            try:
                return brentq(difference_function, search_range[0], search_range[1])
            except ValueError:
                return np.nan # Return NaN if root not found
        # Crossover low (0) vs high (1)
        search_range = (means[0], means[1])
        cutoff = find_crossover(0, 1, search_range)
        if plot:
            if detailed:
                #plotting function detailed
                x_min, x_max = np.min(risk_scores_array), np.max(risk_scores_array)
                x_range = np.linspace(x_min, x_max, 500)
                plt.figure(figsize=figsize)
                plt.hist(risk_scores_array, bins=bins, density=True, alpha=0.6, color='skyblue', label='Risk Score Histogram')
                gmm_total_density = np.zeros_like(x_range)
                for i in range(2):
                    pdf = gmm_pdf(x_range, i)
                    gmm_total_density += pdf
                    plt.plot(x_range, pdf, linestyle='--', label=f'Component {i+1} PDF (Mean: {means[i]:.0f})')
                plt.plot(x_range, gmm_total_density, color='blue', linewidth=1, label='Total GMM Density')
                if not np.isnan(cutoff):
                    plt.axvline(cutoff, color='black', linestyle='-', label=f'Cutoff ({cutoff:.2f})')
                plt.title('GMM Fit')
                plt.xlabel('Risk Score')
                plt.ylabel('Density')
                if legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                if save:
                    import matplotlib as mpl
                    mpl.rcParams['pdf.fonttype'] = 42
                    plt.savefig(save, bbox_inches='tight')
                plt.show()
            
            ###SCORPIO STYLE
            else:
                plt.figure(figsize=figsize)
                x_min, x_max = np.min(risk_scores_array) - 0.05, np.max(risk_scores_array) + 0.05
                x_range = np.linspace(x_min, x_max, 500)
                gmm_total_density = np.zeros_like(x_range)
                for nr, i in enumerate(range(2)):
                    color = "#127812" if nr == 0 else "#951111"
                    pdf = gmm_pdf(x_range, i)
                    gmm_total_density += pdf
                    plt.plot(x_range, pdf, linestyle='--', color=color, label=f'Component {i+1} PDF (Mean: {means[i]:.0f})')
                
                if legend:
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', frameon=False)
                mask_low = x_range <= cutoff
                mask_high = x_range > cutoff
                color_low = '#2ca02c'   
                color_high = '#d62728'  

                plt.fill_between(x_range[mask_low], gmm_total_density[mask_low], color=color_low, alpha=0.5)
                plt.fill_between(x_range[mask_high], gmm_total_density[mask_high], color=color_high, alpha=0.5)

                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.ylabel('Density')
                plt.margins(y=0)

                plt.xlim(x_min, x_max)
                plt.xlabel('Risk Score')

                if save:
                    import matplotlib as mpl
                    mpl.rcParams['pdf.fonttype'] = 42
                    plt.savefig(save, bbox_inches='tight')
                
                plt.show()
            
        self.cutoff_ = cutoff
        return print(f'cutoff between high and low risk calculated at: {cutoff:.2f}.')
  
    #predict using fitted models. generate a average risk score
    def predict(self, X, classify = False, groups = None, verbose = True):
        '''
        Predict risk scores for new samples using the fitted models
        Args:
            X (pd.DataFrame): DataFrame of features for new samples
            classify (bool): Whether to classify risk scores into risk groups. Default is False.
            groups (int): 2 or 3 to specify number of risk groups when classify is True. Default is None.
            verbose (bool): Whether to print messages during prediction. Default is True.
        '''
        if not self.fitted_:
            raise RuntimeError("You must fit the model before predicting.")
        #check if there are missing columns in X, in the case use 0 for missing columns
        missing_cols = set(self.coefs_.index) - set(X.columns)
        if len(missing_cols) > 0:
            if verbose:
                print(f"Missing columns in test compared to training: {missing_cols}")
            for col in missing_cols:
                X[col] = 0
            #put in the same order as training
            X = X.reindex(columns=self.coefs_.index, fill_value=0)

        #check if some columns in X are not in training, in that case drop them
        extra_cols = set(X.columns) - set(self.coefs_.index)
        if len(extra_cols) > 0:
            if verbose:
                print(f"Dropping extra columns in test not seen in training, n. columns: {len(extra_cols)}")
            X = X.drop(columns=extra_cols)
            #put in the same order as training
            X = X.reindex(columns=self.coefs_.index, fill_value=0)
        
        #reorder columns to match training
        X = X.reindex(columns=self.coefs_.index)
        
        #check if some patients have na values
        if X.isnull().values.any():
            if verbose:
                print("Warning: Some samples have missing values. Putting 0 for missing values.")
            X = X.fillna(0)

        risk_scores = []
        for i, model in enumerate(self.models_):
            risk = model.predict(X)
            risk_scores.append(pd.DataFrame({'index': X.index, f'risk_score model {i + 1}': risk}).set_index('index'))
        risk_scores_matrix = pd.concat(risk_scores, axis = 1)
        risk_scores_avg = risk_scores_matrix.mean(axis = 1, skipna=True)
        risk_scores_avg.name = 'risk_score'

        risk_scores_avg = risk_scores_avg.clip(self.train_risk_min_, self.train_risk_max_)
        denom = (self.train_risk_max_ - self.train_risk_min_)
        if denom == 0:
            risk_scores_avg = risk_scores_avg * 0 + 0
        else:
            risk_scores_avg = ((risk_scores_avg - self.train_risk_min_) / denom * 100.0).clip(0, 100)

        if classify:
            if verbose:
                print(f'Activated the classify option, returning risk groups instead of continuous risk scores.')
            if not hasattr(self, 'cutoff_') and not (hasattr(self, 'cutoff_low_intermediate_') and hasattr(self, 'cutoff_intermediate_high_')):
                raise RuntimeError("You must use the define_2_risk_groups or define_3_risk_groups method before classifying.")
            if groups == 3:
                risk_groups = pd.Series(index=risk_scores_avg.index, dtype='object')
                risk_groups = ['high' if score >= self.cutoff_intermediate_high_ else 'intermediate' if score >= self.cutoff_low_intermediate_ else 'low' for score in risk_scores_avg]
                return pd.Series(risk_groups, index=risk_scores_avg.index, name='risk_group')
            elif groups == 2:
                risk_groups = pd.Series(index=risk_scores_avg.index, dtype='object')
                risk_groups = ['high' if score >= self.cutoff_ else 'low' for score in risk_scores_avg]
                return pd.Series(risk_groups, index=risk_scores_avg.index, name='risk_group')
            else:
                raise ValueError("`groups` parameter must be '2' or '3' when classify is True.")
        else:
            if verbose:
                print(f'The option `classify` is set to False, returning continuous risk scores.')
            return risk_scores_avg
    
    #plot selection frequency
    def plot_selection_frequency(self, save=False, label_frequency=0.5, label_coef=0.01, label_coef_low = None, figsize=(6,5), exponentiate = False, despine = False):
        '''
        Plot selection frequency vs mean coefficient
        Args:
            save (str or bool): If str, path to save the figure. If False, do not save. Default is False.
            label_frequency (float): Frequency threshold to label features. Default is 0.5.
            label_coef (float): Coefficient frequency threshold to label features. Default is 0.01.
            label_coef_low (float or None): Lower coefficient frequency threshold to label features if selected exponentiate = True. Default is None.
            figsize (tuple): Figure size. Default is (10,8).
            exponentiate (bool): Whether to exponentiate the mean coefficients (to get hazard ratios). Default is False.
        '''
        import matplotlib.pyplot as plt
        from adjustText import adjust_text
        import seaborn as sns
        import matplotlib as mpl
        plt.figure(figsize=figsize)
        final_coefs = pd.DataFrame({
            'mean_coef': self.coef_mean_,
            'nr_nonzero': (self.coefs_ != 0).sum(axis=1)
        })
        if exponentiate:
            final_coefs['mean_coef'] = np.exp(final_coefs['mean_coef'])
        if exponentiate:
            if label_coef_low is None:
                raise ValueError("When using exponentiate=True, you must provide `label_coef_low` to define lower threshold for labeling.")

        #color as green if coef is negative or if exponentiated coef < 1, else red if positive or > 1, else light gray if below label frequency
        threshold = self.n_models * label_frequency
        def categorize(row):
            if row['nr_nonzero'] <= threshold:
                return 'Low Frequency'
            if exponentiate:
                return 'Positive' if row['mean_coef'] > 1 else 'Negative'
            else:
                return 'Positive' if row['mean_coef'] > 0 else 'Negative'

        final_coefs['color'] = final_coefs.apply(categorize, axis=1)

        if label_coef is not None:
            #put as `Low Frequency` those with abs(coef) < label_coef, and label_coef_low < abs(coef) < label_coef when exponentiated 
            def categorize_with_coef(row):
                if exponentiate:
                    if row['mean_coef'] < label_coef and row['mean_coef'] > label_coef_low:
                        return 'Low Frequency'                        
                else:
                    if abs(row['mean_coef']) < label_coef:
                        return 'Low Frequency'
                return row['color']
            final_coefs['color'] = final_coefs.apply(categorize_with_coef, axis=1)

        # 3. Define the explicit color map
        color_palette = {
            'Positive': '#99000d',
            'Negative': 'green',
            'Low Frequency': 'lightgray'
        }

        sns.scatterplot(data=final_coefs, x='mean_coef', y='nr_nonzero', legend = False, edgecolor='black', linewidth=0.5, hue='color', palette=color_palette)
        if exponentiate:
            plt.axvline(1, color='black', linestyle='--', linewidth=0.6)
        else:
            plt.axvline(0, color='black', linestyle='--', linewidth=0.6)
        texts = []
        for i in range(final_coefs.shape[0]):
            if exponentiate:
                if final_coefs.iloc[i]['nr_nonzero'] > self.n_models * label_frequency and (label_coef is None or abs(final_coefs.iloc[i]['mean_coef']) >= label_coef or abs(final_coefs.iloc[i]['mean_coef']) <= label_coef_low):
                    t = plt.text(final_coefs.iloc[i]['mean_coef'], 
                                final_coefs.iloc[i]['nr_nonzero'], 
                                final_coefs.index[i], 
                                fontsize=10)
                    texts.append(t)
            else:
                if final_coefs.iloc[i]['nr_nonzero'] > self.n_models * label_frequency and (label_coef is None or abs(final_coefs.iloc[i]['mean_coef']) >= label_coef):
                    t = plt.text(final_coefs.iloc[i]['mean_coef'], 
                                final_coefs.iloc[i]['nr_nonzero'], 
                                final_coefs.index[i], 
                                fontsize=10)
                    texts.append(t)
        adjust_text(texts, expand_text=(1.2, 1.2),expand_points=(1.1, 1.1), force_text=(0.5, 0.5), arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
        plt.yticks(np.arange(0, self.n_models+1, max(1, self.n_models//10)))
        plt.ylabel('Selection frequency', fontweight='bold')
        plt.xlabel('Mean coefficient', fontweight='bold')
        if despine:
            sns.despine()
        plt.axes
        if exponentiate:
            plt.xlabel('Hazard Ratio', fontweight='bold')
        plt.tick_params(left = True, bottom = True)
        plt.tight_layout()
        mpl.rcParams['pdf.fonttype'] = 42
        if save:
            plt.savefig(save)
        plt.show()
    
    #plot selection frequency, but for the coefficient magniture just consider non-zero values
    def plot_selection_frequency_nonzero_coef(self, save=False, label_frequency=0.5, figsize=(10,8), exponentiate = False):
        '''
        Plot selection frequency vs mean coefficient (considering only non-zero coefficients for mean calculation)
        Args:
            save (str or bool): If str, path to save the figure. If False, do not save. Default is False.
            label_frequency (float): Frequency threshold to label features. Default is 0.5.
            figsize (tuple): Figure size. Default is (10,8).
            exponentiate (bool): Whether to exponentiate the mean coefficients (to get hazard ratios). Default is False.
        '''        
        import matplotlib.pyplot as plt
        from adjustText import adjust_text
        import seaborn as sns
        import matplotlib as mpl
        plt.figure(figsize=figsize)
        coefs_nonzero = self.coefs_.replace(0, np.nan)
        if exponentiate:
            coefs_nonzero = np.exp(coefs_nonzero)
        mean_nonzero_coefs = coefs_nonzero.mean(axis=1, skipna=True)
        std_nonzero_coefs = coefs_nonzero.std(axis=1, skipna=True)
        final_coefs = pd.DataFrame({
            'mean_coef': mean_nonzero_coefs,
            'nr_nonzero': (self.coefs_ != 0).sum(axis=1), 
            'std_coef': std_nonzero_coefs
        })
        scatter = plt.scatter(final_coefs['mean_coef'], final_coefs['nr_nonzero'], c=final_coefs['std_coef'], cmap='copper', edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, label='Std of Non-zero Coefficients', shrink = 0.25)
        if exponentiate:
            plt.axvline(1, color='black', linestyle='--', linewidth=0.6)
        else:
            plt.axvline(0, color='black', linestyle='--', linewidth=0.6)
        texts = []
        for i in range(final_coefs.shape[0]):
            if final_coefs.iloc[i]['nr_nonzero'] > self.n_models * label_frequency:
                t = plt.text(final_coefs.iloc[i]['mean_coef'], 
                            final_coefs.iloc[i]['nr_nonzero'], 
                            final_coefs.index[i], 
                            fontsize=10)
                texts.append(t)
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
        plt.yticks(np.arange(0, self.n_models+1, max(1, self.n_models//10)))
        plt.grid(axis='y', linestyle='--', linewidth=0.7)
        plt.ylabel('Selection frequency', fontweight='bold')
        plt.xlabel('Mean coefficient', fontweight='bold')
        if exponentiate:
            plt.xlabel('Hazard Ratio', fontweight='bold')

        plt.tight_layout()
        mpl.rcParams['pdf.fonttype'] = 42
        if save:
            plt.savefig(save)
        plt.show()

    
    def plot_train_scores(self, save=False, figsize=(5,3), bins = 30, color = 'blue'):
        import matplotlib as mpl
        plt.figure(figsize=figsize)
        sns.histplot(self.risk_scores_train, bins=bins, color=color)
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        plt.title('Distribution of Training Risk Scores', fontweight='bold')
        plt.tight_layout()
        mpl.rcParams['pdf.fonttype'] = 42
        if save:
            plt.savefig(save)
        plt.show()
    
    #return the risk scores of the train in 3 risks based on GMM
    def get_train_risk_groups_3(self):
        if not self.fitted_:
            raise ValueError("You must fit the model before predicting")
        if not hasattr(self, 'cutoff_low_intermediate_') or not hasattr(self, 'cutoff_intermediate_high_'):
            raise ValueError("You must define 3 risk groups before getting train risk groups with the 'define_3_risk_groups' method.")
        risk_groups = pd.Series(index=self.risk_scores_train.index, dtype='object')
        risk_groups = ['high' if score >= self.cutoff_intermediate_high_ else 'intermediate' if score >= self.cutoff_low_intermediate_ else 'low' for score in self.risk_scores_train]
        return pd.Series(risk_groups, index=self.risk_scores_train.index, name='risk_group')
    
    
    #return the risk scores of the train in 2 risks based on GMM
    def get_train_risk_groups_2(self):
        if not self.fitted_:
            raise ValueError("You must fit the model before predicting with the 'define_2_risk_groups' method.")
        if not hasattr(self, 'cutoff_'):
            raise ValueError("You must define 2 risk groups before getting train risk groups.")
        risk_groups = pd.Series(index=self.risk_scores_train.index, dtype='object')
        risk_groups = ['high' if score >= self.cutoff_ else 'low' for score in self.risk_scores_train]
        return pd.Series(risk_groups, index=self.risk_scores_train.index, name='risk_group')
    
    #calculate cindex on train set
    def calculate_train_cindex(self, y):
        '''
        Calculate Harrel's c-index on the training set using the risk scores from internal validation
        Args:
            y (pd.DataFrame): DataFrame with columns 'Event' and 'Time'
        Returns:
        '''
        if not self.fitted_:
            raise RuntimeError("You must fit the model before calculating c-index.")
        #check if y has column Event and Time
        if 'Event' not in y.columns or 'Time' not in y.columns:
            raise ValueError("y must have columns 'Event' and 'Time', one of the two not found")
        # get common indices between y and risk_scores_train, since may not be included in the outer validation set during model training
        common_idx = y.index.intersection(self.risk_scores_train.index)
        y_surv = Surv.from_dataframe('Event', 'Time', y.loc[common_idx])
        cindex = concordance_index_censored(
            y_surv['Event'], 
            y_surv['Time'], 
            self.risk_scores_train.loc[common_idx]
        )[0]
        return cindex

    def save_model(self, filepath):
        """Saves the current instance to a file using joblib."""
        import joblib
        if not self.fitted_:
            print("Warning: You are saving an unfitted model.")
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Loads a saved instance from a file."""
        import joblib
        return joblib.load(filepath)


    