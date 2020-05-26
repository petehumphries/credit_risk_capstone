# Consumer Loan Credit Risk, Expected Losses and Capital 

In this Flatiron Capstone Project Expected Loss and Capital Requirement are calculated for Lending Club, a Peer to Peer lender, using customer loans data to estimate the three target variables Probability of Default, Exposure at Default and Loss Given Default that make up Expected Loss. Credit Scorecards are created alone the way to assess new loan applications. 

Credit Risk Models are used to decide who to make loans to, how to manage the risk of those loans, and how much capital the bank should set aside as a buffer against shocks .

### Executive Summary

Peer to Peer lenders have a limited amounts of capital, and use credit risk models to allocate their capital efficiently as possible when deciding which loans applications to approve and which ones to decline.  The goal is to increase approval rates for good borrowers, and decrease the approval rates for bad borrowers, using a scorecard as a neutral, objective way to assess credit applications.

To help solve this problem a combination of models were used to estimate the component parts of expected loss. A classification model was used to estimate the Probability of Default for individual loans, an estimation model was used to estimate customer Exposure at Default and a combination of the classification and estimation models were used to estimate Loss Given Default.  These components were used to create a customer scorecard, decision rules and finally calculate the capital requirement.   

A loans dataset containing customer loans information from Lending Club for 2007-2015 was sourced from Kaggle, and independent macro economic data was sourced from macro trends which is great information source. Feel free to check them out at: https://www.macrotrends.net/!

There are many versions of the Lending CLub Dataset but went with the original:

* Lending Club Loans Dataset #1 https://www.kaggle.com/wendykan/lending-club-loan-data/version/1?
* For NATIONAL UNEMPLOYMENT do follow: https://www.macrotrends.net/1316/us-national-unemployment-rate
* For RETAIL SALES do follow:https://www.macrotrends.net/1371/retail-sales-historical-chart
* For TED SPREAD do follow: https://www.macrotrends.net/1447/ted-spread-historical-chart
* For VIX do follow: https://www.macrotrends.net/2603/vix-volatility-index-historical-chart

TIn order to estimate Probability of Default a Logistic Regression with 25 inputs was used to categorise obligors into good borrowers and bad borrowers. Features were split into similar classes and features with poor explanatory power were dropped in a feature selection prior to running the baseline model generating an AUROC of 70.29%. Further runs of the model with fewer inputs look promising as it appears that simple is better for PD models.

For the LGD modeling accuracy is 0.6083 with AUROC 66.78% and mean Exposure at Default is 73.16%.

The average Expected Loss on loans in the dataset is 5.6129%.

The most important files in the repo are:

  * notebooks/credit_risk_modeling_03_modeling.ipynb - main notebook for modelling and insights
  * notebooks/credit_risk_modeling_02_preparation.ipynb - main notebook for fine classing of variables 
  * data - folder containing all data sources, raw and cleaned


last bit: could you add the "Future improvements" section to your readme as well? You had them in your presentation so it should be a straightforward copy and paste. Just add them at the end of the readme and they'll nicely finish it

### Visualizations

![WoE by state](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/WoE_by_state.JPG)
![WoE by income](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/WoE_by_income.JPG)
![WoE by interest_rate](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/WoE_by_interest_rate.JPG)
![WoE by payment_to_income](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/WoE_by_payment_to_income_factor.JPG)
![WoE by subgrade](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/WoE_by_subgrade.JPG)
![garp_logo](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/garp_logo.JPG)
![flatiron_logo](https://github.com/petehumphries/flatiron_capstone_final/blob/master/images/flatiron_logo.JPG)


For Anyone Interested in Risk Management or Credit Risk the Global Association of Risk Professional's FRM programme is worth checking out:

https://www.garp.org/?gclid=EAIaIQobChMIqoDgpfrE6QIV1GDmCh1OMg_DEAAYASAAEgL27vD_BwE#!/frm
# credit_risk_capstone
