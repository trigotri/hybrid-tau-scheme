This git repository contains the code used to generate the data and figures in "A hybrid tau-leap for simulating chemical kinetics with applications to parameter estimation". 

Some of the codes used to generate ensembles of realisations were executed on the [Eddie research compute cluster](https://www.ed.ac.uk/information-services/research-support/research-computing/ecdf/high-performance-computing), and the resulting data re-assembled and post-processed afterwards. 

Those files are then typically executed as 
```python
python num_realisations proc_id xp_arg1 xp_arg2 ...
```

The remaining files were run on a laptop, and can be run as
```python filename.py
after commenting in/out the relevant functions in ```__main__``` part of the script. 
