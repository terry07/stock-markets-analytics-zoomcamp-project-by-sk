# 💸💸💸💸top-24 stocks investment simulation project📢
As a participant in the Stock Market Analytics Zoomcamp of 2025, we are required to deliver a final project that applies everything we have learned in the course to build an end-to-end machine learning pipeline. The ultimate goal is to run simulations that allow us to evaluate investment strategies before applying them in practice.

- `Course link`: https://pythoninvest.com/course
- `Github repo`: https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/tree/main


## Dependencies Installation

Create new Python environment for installing the necessary dependencies. The suggested and verified steps are the following, using conda environment:

```bash
conda create -n <YOUR-PREFERRED-NAME> python==3.12
```

After that, install the underlying dependencies through the next command:
```bash
pip install -r requirements.txt
```

In order to enable the `pre-commit' rules, run the next command (the installation has already applied in the above command):
```bash
pre-commit install
```

If `pre-commit` is enabled, you will see output from the hooks (ruff, black, isort) running before the commit is finalized. If there are issues, the commit will be blocked until you fix them.

If you want to manually run `pre-commit`, e.g., for including files that had been oushed before setting that functionality, run the next command:

```bash
pre-commit run --all-files
```
