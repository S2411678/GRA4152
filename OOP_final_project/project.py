import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps
from scipy.optimize import minimize
import argparse
import textwrap

# create help for GLM superclass and subclasses
parser = argparse.ArgumentParser(
    prog = 'GLM superclass and subclasses',
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = textwrap.dedent('''\
            GLM Superclass
        --------------------------------
        A simulated unified class of models for regression analysis of independent
        observations of a discrete or continuous response : Normal, Bernoulli, and Poisson

        Abstract methods: 
        Implementation of abstract methods specified in each subclass

        1) ne_loglik: defines the negative log-likelihood function, essential for model fitting 

        2) predict: return the expected value of dependent variable Y                              

        Concrete methods:
        1) mle: estimate parameter beta of the model by maximizing log-likelihood function

        2) params (@property): an accessor method which retrieves the parameter beta
        '''),
    epilog = textwrap.dedent('''\
            Usage Example
        --------------------------------
        model = Normal(X,Y) # initialize an object
        model.mle() # fit the model
        model.params # access beta values
        model.predict(X) # return expected value of Y
        ''')
)


class DataLoader:

    def load_data(self):
        raise NotImplementedError    
        
    def split(self, xcolumns, ycolumn):
        # define the x and y columns
        # self._data loaded after load_data method
        # hence self._data should be defined in subclass
        # avoid creating empty instance variable
        # if X contains many columns then xcolumns must be provided as a list                
        self._y = self._data[ycolumn]
        self._x = self._data[xcolumns]
    
    @property
    def values(self):
        return self._data
        
    @property
    def xvalues(self):        
        return self._x
    
    @property
    def yvalues(self):        
        return self._y        
    
    def add_const(self):
        # using built-in add_constant in statsmodels
        # alter name to distinguish with built-in function
        assert 'const' not in self._x.columns, "Already add_constant"
        self._x = sm.add_constant(self._x)
        return self._x         
    
    def xtranspose(self):
        # this coule be achieved by transpose method in numpy
        # alter name to distinguish with built-in method
        return self._x.transpose()
    
class rDataLoader(DataLoader):
# fetch dataset from R repository datasets

    def __init__(self, package, dataset):        
        self._package = package
        self._dataset = dataset

    def load_data(self):        
        try:
            self._data = sm.datasets.get_rdataset(self._dataset, package = self._package).data
            print(self._data.info()) # summary dataset            
        except Exception as e:
            print(e)

class smDataLoader(DataLoader):
# load data from built-in statsmodel datasets : spector, star98 ...
    
    def __init__(self, dataset):        
        self._dataset = dataset

    def load_data(self):
        try:
            self._data = getattr(sm.datasets, self._dataset).load_pandas().data
            print(self._data.info()) # summary dataset
            return self._data
        except Exception as e:
            print(e)

class locDataLoader(DataLoader):
# load .csv file from local computer
    
    def __init__(self, path):
        self._loc = path

    def load_data(self):        
        try:
            self._data = pd.read_csv(self._loc)
            print(self._data.info()) # summary dataset
        except Exception as e:
            print(e)

class iDataLoader(DataLoader):
# load .csv data from the Internet
    def __init__(self, url):
        self._url = url

    def load_data(self):
        # some csv file online use 'tab' or ';' to seperate data columns        
        # data will be read into one column
        # Example: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
        # hence several options for handling different separators
        # usually read_csv use C engine 
        # but since multiple separators are specified
        # engine = python is required to read data
        # however this will reduce the speed of parse data
        # trade off between simplicity, time efficiency and automatic
        try:
            self._data = pd.read_csv(self._url, sep=r'\,|\t|\;', engine='python')
            print(self._data.info()) # summary dataset
        except Exception as e:
            print(e)

class GLM:

    def __init__(self, x = None, y = None):
        self._x = x
        self._y = y
        self._parser = parser

    @property
    def help(self):
        self._parser.print_help()

    def ne_loglik(self, params, x, y):
        raise NotImplementedError 

    def mle(self): 
        # same responsibility as 'fit' in statsmodel
        num_params = self._x.shape[1]
        init_params = np.repeat(0.1, num_params)
        results = minimize(self.ne_loglik, init_params, args = (self._x, self._y))
        self._params = results['x']         
        print('optimization is finished\n', self._params)                   
                       
    def predict(self, x_new):      
        raise NotImplementedError
    
    @property
    def params(self):
        return self._params

class Normal(GLM):

    def __init__(self, x = None, y = None):
        super().__init__(x, y)

    def ne_loglik(self, params, x, y):
        eta = x @ params
        mu = eta        
        ne_llik = -np.sum(sps.norm.logpdf(y, mu))
        return ne_llik 
    
    def predict(self, x_new):
        eta = x_new @ self._params
        mu = eta
        return mu

class Bernoulli(GLM):

    def __init__(self, x = None, y = None):
        super().__init__(x, y)

    def ne_loglik(self, params, x, y):  
        eta = x @ params             
        mu = 1/(1 + np.exp(-eta))
        ne_llik = -np.sum(sps.bernoulli.logpmf(y, mu))
        return ne_llik    
        
    def predict(self, x_new):
        eta = x_new @ self._params
        mu = 1/(1 + np.exp(-eta))
        return mu   

class Poisson(GLM):

    def __init__(self, x = None, y = None):
        super().__init__(x, y)

    def ne_loglik(self, params, x, y):
        eta = x @ params
        mu = np.exp(eta)       
        ne_llik = -np.sum(sps.poisson.logpmf(y, mu))
        return ne_llik
    
    def predict(self, x_new):
        eta = x_new @ self._params
        mu = np.exp(eta)
        return mu
    
if __name__ == "__main__":
    args = parser.parse_args()