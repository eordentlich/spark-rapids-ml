## Running on Databricks AWS
- Follow instructions [here](../databricks/README.md) to get started with spark-rapids-ml on Databricks
- To run the examples here, including the data generation scripts
  - Upload these notebooks to a folder in your workspace.
  - Upload the files [gen_data.py](../../python/benchmark/gen_data.py) and [gen_data_distributed.py](../../python/benchmark/gen_data_distributed.py) to the same workspace folder as the notebooks.
  - Upload the following from [benchmark](../../python/benchmark/benchmark) to a folder `benchmark` matching the below structure, also in the above workspace folder.
    ```
    benchmark/
       __init__.py
       utils.py
    ```
  - The cluster info and configs for the GPU cluster are in the file [databricks_cluster_info.json](databricks_cluster_info.json) and can be entered manually via the Databricks cluster creation UI.
  - For the CPU run one of `m5dn.4xlarge`, `m6idn.4xlarge`, `m7gd.4xlarge` executor instance types used.
  - Note that in comparing GPU and CPU runs, different numbers of iterations may have been carried out before stopping criteria were reached.
  - Also GPU runs are single precision float by default, while Spark CPU is double precision.
