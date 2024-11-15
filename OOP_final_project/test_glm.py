import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps
from scipy.optimize import minimize

from project import iDataLoader, smDataLoader, rDataLoader, Normal, Bernoulli, Poisson

def tester(source, distribution):
    tol = 10**(-3) # tolerance level to compare beta estimated by class GLM and statsmodels
    if source == "i":
        url = "https://raw.githubusercontent.com/BI-DS/GRA-4152/refs/heads/master/warpbreaks.csv"
        # Create an object that loads .csv file from online source
        iloader = iDataLoader(url)
        iloader.load_data() # load data from source
        # the data follows poisson disttibution
        poisson = iloader.values
        print(f"\nExample data in the dataset:\n",poisson.tail())
        # Choose columns for X and Y
        iloader.split(['wool','tension'],'breaks')
        # assign xvalues to X and yvalues Y as inputs for an object of subclass Poisson of GLM class        
        Y = iloader.yvalues
        X = iloader.xvalues
        print(f"\nExample of X values:\n",X.tail(),
            '\nExample of Y values:\n',Y.tail())
        # add_constant to xvalues
        # add_constant the second time will return AssertionError: Already add_constant
        X = iloader.add_const()
        print(f"\nExample of add constants to X values:\n",X.tail())
        # tranpose X for matrix multiplication
        print(f"\nExample of transpose X values:\n",
            iloader.xtranspose().iloc[:,-6:])
        

    if source == "r":
        name = "Duncan"
        pack = "carData"
        # Create an object that loads data from R datasets
        rloader = rDataLoader(pack, name)        
        rloader.load_data()
        # the data follows normal disttibution
        normal = rloader.values
        print(f"\nExample data in the dataset:\n",normal.head())
        # Choose columns for X and Y
        rloader.split(['education', 'prestige'],'income')
        # assign xvalues to X and yvalues Y as inputs for an object of subclass Normal of GLM class
        Y = rloader.yvalues
        X = rloader.xvalues
        print(f"\nExample of X values:\n",X.head(),
            '\nExample of Y values:\n',Y.head())
        # add_constant to xvalues
        # add_constant the second time will return AssertionError: Already add_constant
        X = rloader.add_const()
        print(f"\nExample of add constants to X values:\n",X.head())
        # tranpose X for matrix multiplication
        print(f"\nExample of transpose X values:\n",
            rloader.xtranspose().iloc[:,:5])
        

    if source == "sm":
        name = 'spector'
        # Create an object that loads data from datasets inside statsmodels
        smloader = smDataLoader(name)
        smloader.load_data()
        # the data follows bernoulli disttibution
        bernoulli = smloader.values
        print(f"\nExample data in the dataset:\n",bernoulli.head())
        # Choose columns for X and Y
        smloader.split(['GPA','TUCE','PSI'],'GRADE')
        # assign xvalues to X and yvalues Y as inputs for an object of subclass Bernoulli of GLM class
        Y = smloader.yvalues
        X = smloader.xvalues
        print(f"\nExample of X values:\n",X.head(),
            '\nExample of Y values:\n',Y.head())
        # add_constant to xvalues
        # add_constant the second time will return AssertionError: Already add_constant
        X = smloader.add_const()
        print(f"\nExample of add constants to X values:\n",X.head())
        # tranpose X
        print(f"\nExample of transpose X values:\n",
            smloader.xtranspose().iloc[:,:5])
        
    model_selection = {'normal': Normal, 'bernoulli': Bernoulli, 'poisson': Poisson}
    
    model = model_selection[distribution]
    fitmodel = model(X,Y) # pass data into the model
    # use 'mle' to fit the model
    fitmodel.mle()
    
    pred = {'normal': [1,90,80], 
            'bernoulli': [1,3.3,20,1],
            'poisson': [1,0,2]}

    # use 'predict' to predict expected value of Y
    print(f"\nPredict y-value based on x-value {pred[distribution]}:\n", 
          fitmodel.predict(pred[distribution]))
    
    # Fit beta using built-in function of statsmodels
    # For comparison with results from GLM class
    if distribution == "normal":
        model_sm = sm.GLM(Y, X, family = sm.families.Gaussian()).fit()
    elif distribution == "bernoulli":
        model_sm = sm.GLM(Y, X, family = sm.families.Binomial()).fit()
    elif distribution == 'poisson':
        model_sm = sm.GLM(Y, X, family = sm.families.Poisson()).fit()
    else:
        print("Not available model")

    # print the beta and predicted value obtained from statsmodels built-in function
    print(f"beta estimations by built-in function:\n",
                model_sm.params)
    print(f"\nPredict y-value by built-in function:\n",
           model_sm.predict(pred[distribution]))
    # Compare beta
    print(f"\nCompare the results:\n",
                abs(fitmodel.params - model_sm.params) < tol)

    
if __name__ == '__main__':
    tester('sm','bernoulli')

### Example main arguments: ('sm','bernoulli') - ('i','poisson') - ("r","normal")