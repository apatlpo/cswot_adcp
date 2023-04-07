# cswot_adcp

Quick scripts and notebooks to inspect adcp data.

## install

Environment required my be installed with [miniconda](https://docs.conda.io/en/latest/miniconda.html):

```
git clone https://github.com/apatlpo/cswot_adcp.git
cd cswot_adcp
conda create -n cswot python=3.10
conda env update -n cswot -f environment.yml
conda activate cswot
```

Install finally the cswot library with `pip install -e .` from the root directory (`cswot_adcp`).

## useful scripts and commands

- to launch the user interface and inspect an `.sta` file, execute from `cswot_adcp/sandbox` the following command:

```python
python main_app.py --amplitude_max .5 --correlation_max 256 "/Volumes/missioncourante/adcp/TT-OS75-2021-388_000000.STA"
```

- to convert sta files to netcdf:

```python
python convert2nc.py /Volumes/missioncourante/adcp/TT-OS75-2021-3*_000000.STA
```

