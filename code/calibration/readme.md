# Calibration Code
for running abc-smc calibration
## Files Description
- `baseline_model_single_strain2_data2.py` and `baseline_model_two_strains2_data2.py` are for baseline model
- `behaviour_model_single_strain2_data2.py` and `behaviour_model_two_strains2_data2.py` are for behavioural models 1 and 2
- `variation_model_single_strain2_data2.py` and `variation_model_two_strains2_data2.py` are for behavioural models 3 and 4

## Running the Code
The main script (`main.py`) for running the calibration process is not included in this repository. To run it, please create a script with the following content:

```python
from read_params import run_abc

if __name__ == "__main__":
    run_abc(region, model_id)
```
input for region can be 'British Columbia' or 'Lombardy' or 'London' or 'Sao Paulo'
input for model_id can be 'baseline', 'model1', 'model2', 'model3', 'model4'
