
**QueryBasedVerification** is a defense module implemented under the **PyGIP** framework that replicates the core defense proposed in the paper _"Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based Integrity Verification"_ (Wu et al., 2023).


## Experimental Parameters**

#### Common Parameters
| Parameter              | Value Used                               | Paper Value                                | Notes                                                           |
| ---------------------- | ---------------------------------------- | ------------------------------------------ |
| `attack_node_fraction` | `0.1`                                    | `0.3`                                      | Lowered to reduce impact and runtime                            |
| `k` (num fingerprints) | `5`                                      | `10`                                       | Halved to reduce query overhead while maintaining effectiveness |
| `attack_trial_map`     | `bitflip: 20`, `random: 5`, `mettack: 5` | Paper uses 400 trials for BFA              | Reduced for faster experimentation                              |
| `bit_position`         | `30`                                     | Unspecified, but paper flips exponent bits | Matches intent of BFA attack                                    |

#### Bit Flip Attack (BFA) Specific Parameters

| Parameter     | Value Used              | Paper Value | Notes                          |
| ------------- | ----------------------- | ----------- | ------------------------------ |
| `num_trials`  | `20`                    | `400`       | Downsampled to speed up runs   |

#### Random Poisoning Attack

| Parameter     | Value Used | Paper Value               | Notes                      |
| ------------- | ---------- | ------------------------- | -------------------------- |
| `num_trials`  | `5`        | Not directly specified    | Chosen for time-efficiency |

#### Mettack Poisoning Attack

| Parameter               | Value Used | Paper Value                    | Notes                                           |
| ----------------------- | ---------- | ------------------------------ | ----------------------------------------------- |
| `poison_frac`           | `0.005`    | ~0.01 (for 100 perturbations)  | Halved to ~50 perturbations for faster runtime  |
| `epochs`                | `30`       | `200`                          | Reduced to speed up training                    |
| `surrogate_epochs`      | `20`       | `200`                          | Reduced for surrogate model efficiency          |
| `candidate_sample_size` | `50`       | `100` (default)                | Smaller pool for runtime reasons                |


## Results

**Cora Dataset**

### Transductive-F Detection Rate Comparison
| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.69               | 0.711                |
| BFA-F     | 0.67               | 0.96                 |
| BFA-L     | 0.7                | 0.5                  |
| random    | 0.52               | 0.647                |
| mettack   | 0.84               | 0.588                |

### Transductive-L Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.63               | 0.982                |
| BFA-F     | 0.77               | 0.81                 |
| BFA-L     | 0.74               | 1.0                  |
| random    | 0.72               | 0.353                |
| mettack   | 0.88               | 0.598                |

### Inductive-F Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.58               | 0.667                |
| BFA-F     | 0.7                | 1.0                  |
| BFA-L     | 0.66               | 0.382                |
| random    | 0.68               | 1.0                  |
| mettack   | 1.0                | 1.0                  |

### Inductive-L Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.73               | 0.688                |
| BFA-F     | 0.62               | 0.989                |
| BFA-L     | 0.63               | 0.348                |
| random    | 0.44               | 1.0                  |
| mettack   | 0.68               | 1.0                  |


**Citeseer Dataset**

### Transductive-F Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.63               | 0.586                |
| BFA-F     | 0.70               | 0.430                |
| BFA-L     | 0.56               | 0.529                |
| random    | 0.60               | 0.412                |
| mettack   | 0.68               | 0.353                |

### Transductive-L Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.56               | 0.289                |
| BFA-F     | 0.61               | 0.430                |
| BFA-L     | 0.53               | 0.133                |
| random    | 0.76               | 0.824                |
| mettack   | 0.68               | 0.235                |

### Inductive-F Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.48               | 0.941                |
| BFA-F     | 0.57               | 0.882                |
| BFA-L     | 0.68               | 0.529                |
| random    | 0.64               | 1.0                  |
| mettack   | 0.92               | 1.0                  |

### Inductive-L Detection Rate Comparison

| Attack    | Our Detection Rate | Paper Detection Rate |
|-----------|--------------------|----------------------|
| BFA       | 0.59               | 0.901                |
| BFA-F     | 0.53               | 0.852                |
| BFA-L     | 0.67               | 0.569                |
| random    | 0.72               | 1.0                  |
| mettack   | 0.92               | 1.0                  |
