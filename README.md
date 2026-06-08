# HSRWOD
Hyper-Box-Based Second-Order Biased Random Walk for Unsupervised Outlier Detection


## Data Format

- `.txt` file, space/tab separated, last column is label
- `.mat` → `.txt`: use `dataprocess.py`

## Run

```bash
python main.py
```

## Parameters

- `rho` : Attribute retention ratio (default: 0.5)
- `gamma` : Hyper-box scale factor (default: 1.0)
- `p` : Walk return parameter (0.5 for BFS, 2.0 for DFS)
- `q` : Walk in-out parameter (2.0 for BFS, 0.5 for DFS)

## Code Structure

- `main.py` : Main entry point
- `dataprocess.py` : `.mat` to `.txt` conversion
- `graph_rw.py` : Random walk, graph, fuzzy rough
- `hyperbox.py` : Hyper-box granulation, feature distillation


