Place local data files for the policy notebooks in this folder.

Supported synthetic-control inputs:

1. California cigarette-tax example
   - preferred filenames:
     - `california_prop99.csv`
     - `prop99.csv`
     - `smoking.csv`
   - required columns after import:
     - `State`
     - `Year`
     - `PacksPerCapita`
   - the loader also accepts common aliases such as `state`, `year`, and `cigsale`.

2. German reunification example
   - preferred filenames:
     - `scpi_germany.csv`
     - `germany_reunification.csv`
   - required columns:
     - `country`
     - `year`
     - `gdp`
   - optional columns used when available:
     - `trade`
     - `schooling`
     - `industry`
     - `infrate`

3. Difference-in-differences fatalities panel
   - optional local filenames:
     - `Fatalities.csv`
     - `AER_Fatalities.csv`
   - if these are absent, the notebook falls back to `AER::Fatalities` through `statsmodels`.
