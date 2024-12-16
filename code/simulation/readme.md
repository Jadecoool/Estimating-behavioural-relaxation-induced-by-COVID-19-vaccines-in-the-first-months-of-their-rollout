# Simulation Code
for running simulations
## Files Description
- `cf_vaccine` is for the counterfactual scenario of vaccination
- `cf_mobility` is for the counterfactual scenario of  NPIs
- `cf_behaviour` is for the counterfactual scenario of behavioural mechanisms
- `run_behaviour_record_NC` is for recording the number of non-compliant/compliant individuals

## Running the Code
The main script (`main.py`) for running the simulations is not included in this repository. To run it, please create separate scripts with the following content:

for vaccination counterfactual scenarios:
```python
# This is a generated Python script for region in scenario cf_vaccine
from cf_vaccine import run_cf

if __name__ == "__main__":
    run_cf(region)
```

for NPI counterfactual scenarios:
```python
# This is a generated Python script for region in scenario cf_mobility
from cf_mobility import run_cf

if __name__ == "__main__":
    run_cf(region)
```

for behavioural counterfactual scenarios:
```python
# This is a generated Python script for region in scenario cf_behaviour
from cf_behaviour import run_cf

if __name__ == "__main__":
    run_cf(region, model_id)
```

for recording the number of NC/C individuals:
```python
# This is a generated Python script for region of recording NC/C
from run_behaviour_record_NC import get_NCandC

if __name__ == "__main__":
    get_NCandC(region, model_id)
```

input for region can be 'British Columbia' or 'Lombardy' or 'London' or 'Sao Paulo'
input for model_id can be '_07_' or '_08_' or '_09_' or '_10_'. Note: '_07_' correspond 'model 1', '_08_' correspond 'model 2', '_09_' correspond 'model 3', '_10_' correspond 'model 4'
