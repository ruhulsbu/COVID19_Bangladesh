import pandas as pd
import numpy as np
import requiredFunctions as rf
import logging
from IPython.display import clear_output

fileLoc  = "dataset.xlsx"
sheetLoc = "Sheet1"

def get_data(area_name):
    for column in datasetxl:
        if(column != "Date" and column==area_name):
            dataSeries = pd.Series(datasetxl[column].values, index=datasetxl['Date'])
            pd.to_numeric(dataSeries, errors="coerce")
            R_T_MAX = 12
            r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)
            GAMMA = 1/7
            original, smoothed = rf.prepare_cases(dataSeries)
            posteriors, log_likelihood = rf.get_posteriors(smoothed, GAMMA, r_t_range, sigma=.25)
            sigmas = np.linspace(1/20, 1, 20)
            new, smoothed = rf.prepare_cases(dataSeries, cutoff=25)
            if len(smoothed) == 0:
                new, smoothed = rf.prepare_cases(dataSeries, cutoff=10)
            result = {}
            result['posteriors'] = []
            result['log_likelihoods'] = []
            for sigma in sigmas:
                posteriors, log_likelihood = rf.get_posteriors(smoothed, GAMMA, r_t_range, sigma=sigma)
                result['posteriors'].append(posteriors)
                result['log_likelihoods'].append(log_likelihood)
            clear_output(wait=True)
            sigma = sigmas[np.argmax(result['log_likelihoods'])]
            posteriors = result['posteriors'][np.argmax(result['log_likelihoods'])]
            logging.debug(f"Sigma: {sigma} has highest log likelihood")
            logging.debug('Done.')
            hdis = rf.highest_density_interval(posteriors, p=.9)
            most_likely = posteriors.idxmax().rename('ML')
            result = pd.concat([most_likely, hdis], axis=1)
            return result

datasetxl = rf.prepare_data(fileLoc, sheetLoc)

json = get_data("total").to_json(orient='index')
print(json)