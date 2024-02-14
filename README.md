# BoomBikes: Predictive Analytics for Post-Pandemic Bike-Sharing Demand
> This project aims to deploy multiple linear regression analysis to forecast the demand for shared bikes in the post-COVID-19 landscape. Utilizing a comprehensive dataset, the project will identify key variables influencing bike-sharing demand in the American market, enabling BoomBikes to refine their business strategies. By understanding how factors like weather conditions, holidays, and the day of the week affect bike usage, BoomBikes will be better positioned to meet customer expectations and drive revenue growth in a recovering economy.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Background: This project is designed to address the challenges faced by BoomBikes, a bike-sharing provider in the US, whose revenues have significantly declined due to the COVID-19 pandemic. The service allows users to borrow bikes from docks for short-term use.
- Business Problem: The project aims to predict the demand for shared bikes post-pandemic, helping BoomBikes to strategically plan for revenue acceleration once the economy recovers.
- Dataset: The dataset includes daily data on bike demands across the American market, incorporating various factors such as meteorological conditions and user demographics. 

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
### EDA
- Temperature and 'feels like' temperature are strongly correlated with bike rentals, indicating higher temperatures boost demand.
- Seasonality significantly impacts bike rentals, with certain seasons, likely warmer ones, experiencing higher demand.
- Clear weather conditions favor higher bike rentals, while adverse weather reduces demand.
- The dataset shows a year-over-year increase in bike rentals, suggesting growing popularity or improved infrastructure.

### Modelling
- Developed multiple models, that ended up with 2 final models: one without binning and one with binning.
- First model: R^2 of 83.5% (training), 80.5% (testing); highlights temperature, year, and weather situation as key predictors.
    - Equation of the best fitted line: cnt = 2385.14 + (2044.46 * yr) + (3756.44 * temp) − (1304.85 * windspeed) − (898.52 * season_spring) + (364.72 * season_winter) − (386.75 * mnth_January) − (573.25 * mnth_July) + (465.57 * mnth_September) − (389.82 * weekday_Sunday) − (2504.89 * weathersit_Light Precipitation) − (696.54 * weathersit_Mist)
- Second model: Improved R^2 of 84.3% (training), 82.6% (testing); introduces binned variables for nuanced analysis. It also highlights temperature, year, and weather situation as key predictors.
    - Equation of the best fitted line: cnt = 1758.74 + (1959.55 * yr) + (4047.22 * temp) − (885.74 * season_spring) + (432.54 * season_winter) − (463.05 * mnth_July) + (491.14 * mnth_September) − (334.00 * weekday_Sunday) − (2486.63 * weathersit_Light Precipitation) − (550.44 * weathersit_Mist) + (534.48 * temp_bins_Medium) − (309.54 * hum_bins_High) − (317.70 * windspeed_bins_High)
- Binning in the second model categorizes variables like temperature and humidity, enhancing model performance.
- Best predictors include temperature, season, and clear weather conditions across both models.
- Final conclusion: The second model, with binned variables, outperforms the first in predictive accuracy, indicating a better approach for understanding bike rental demand nuances.

## Recommendations
- Prioritize bike availability and marketing during warmer seasons to capitalize on increased demand.
- Focus on clear weather days for promotional activities, aligning with higher rental trends.
- Adapt strategies based on year-over-year trends to leverage growing bike-sharing popularity.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.11.3
- pandas - version 1.5.3
- numpy - version 1.24.3
- matplotlib - 3.7.1
- seaborn - 0.12.2
- anaconda - 23.5.2

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


## Contact
Created by [@AnirbanG-git] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->