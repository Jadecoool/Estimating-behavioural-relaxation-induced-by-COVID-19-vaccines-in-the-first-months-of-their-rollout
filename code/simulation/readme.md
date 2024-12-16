# Simulation Code
for running simulations
## Files Description
- `cf_vaccine` is for the counterfactual scenario of vaccination
- `cf_mobility` is for the counterfactual scenario of  NPIs
- `cf_behaviour` is for the counterfactual scenario of behavioural mechanisms
- `run_behaviour_record_NC` is for recording the number of non-compliant/compliant individuals

## Running the Code
The main script (`main.py`) for running the simulations is not included in this repository. To run it, please create a script with the following content:

```python
# This is a generated Python script for Lombardy in scenario cf_vaccine
from cf_vaccine import run_cf
import numpy as np

if __name__ == "__main__":
    run_cf(region)
```
```python
# This is a generated Python script for Lombardy in scenario cf_vaccine
from cf_mobility import run_cf
import numpy as np

if __name__ == "__main__":
    run_cf(region)
```
```python
# This is a generated Python script for Lombardy in scenario cf_vaccine
from cf_behaviour import run_cf
import numpy as np

if __name__ == "__main__":
    run_cf(region, model_id)
```
input for region can be 'British Columbia' or 'Lombardy' or 'London' or 'Sao Paulo'
