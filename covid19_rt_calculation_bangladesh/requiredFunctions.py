import pandas as pd
import numpy as np
from scipy import stats as sps

def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed


def prepare_data(fileLoc,sheetLoc):
    datasetxl = pd.read_excel(fileLoc,sheetLoc)
    datasetxl = datasetxl.dropna(how='all', axis='columns')
    datasetxl = datasetxl.loc[:, ~datasetxl.columns.str.contains('^Unnamed',na=False)]
    datasetxl = datasetxl.drop([0,67,68,69])
    datasetxl.to_csv("dataset.csv")
    datasetxl = pd.read_csv("dataset.csv",header=None)
    datasetxl = datasetxl.transpose()
    datasetxl = datasetxl.drop([0,1])
    datasetxl = datasetxl.reindex(index=datasetxl.index[::-1])
    datasetxl.columns=["Date","B. Baria","Bagerhat","Bandarban","Barguna","Barisal","Bhola","Bogra","Chandpur","Chapainawabganj","Chattogram","Chuadanga","Cox’s bazar","Cumilla","Dhaka (District)","Dhaka City","Dinajpur","Faridpur","Feni","Gaibandha","Gazipur","Gopalganj","Habiganj","Jamalpur","Jessore","Jhalokathi","Jhenaidah","Joypurhat","Khagrachhari","Khulna","Kishoreganj","Kurigram","Kushtia","Lakshmipur","Lalmonirhat","Madaripur","Magura","Manikganj","Meherpur","Moulvibazar","Munshiganj","Mymensingh","Naogaon","Narail","Narayanganj","Narsingdi","Natore","Netrokona","Nilphamari","Noakhali","Pabna","Panchagarh","Pirojpur","Potuakhali","Rajbari","Rajshahi","Rangamati","Rangpur","Satkhira","Shariatpur","Sherpur","Sirajganj","Sunamganj","Sylhet","Tangail","Thakurgaon","total"]
    datasetxl['Date'] = pd.to_datetime(datasetxl['Date']).dt.strftime('%Y-%m-%d')
    districts = ['B. Baria','Bagerhat','Bandarban','Barguna','Barisal','Bhola','Bogra','Chandpur','Chapainawabganj','Chattogram','Chuadanga','Cox’s bazar','Cumilla','Dhaka (District)','Dhaka City','Dinajpur','Faridpur','Feni','Gaibandha','Gazipur','Gopalganj','Habiganj','Jamalpur','Jessore','Jhalokathi','Jhenaidah','Joypurhat','Khagrachhari','Khulna','Kishoreganj','Kurigram','Kushtia','Lakshmipur','Lalmonirhat','Madaripur','Magura','Manikganj','Meherpur','Moulvibazar','Munshiganj','Mymensingh','Naogaon','Narail','Narayanganj','Narsingdi','Natore','Netrokona','Nilphamari','Noakhali','Pabna','Panchagarh','Pirojpur','Potuakhali','Rajbari','Rajshahi','Rangamati','Rangpur','Satkhira','Shariatpur','Sherpur','Sirajganj','Sunamganj','Sylhet','Tangail','Thakurgaon','total']
    datasetxl[districts] = datasetxl[districts].fillna(0.0)
    datasetxl[districts] = datasetxl[districts].apply(pd.to_numeric, errors='coerce')
    datasetxl[districts] = datasetxl[districts].cumsum()
    return datasetxl

def get_posteriors(sr, GAMMA, r_t_range, sigma=0.15):

    # (1) Calculate Lambda
    # Map Rt into lambda so we can substitute it into the equation below
    # Note that we have N-1 lambdas because on the first day of an outbreak
    # you do not know what to expect.
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])
    
    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0
    
    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood

#hide_input
# Calculate High density interval
def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]
    
    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()
    
    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]
    
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


