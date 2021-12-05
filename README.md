# Weather cast using Reservoir Computing with LBM

## Get Started

you have to install these packages to your system in advance: python3.9, wgrib2, grib2json

Then, activate the venv environment,

```bash
$ python3.9 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ cp .env.example .env (you should adapt .env file to your environment )
```

when you deactivate venv,

```bash
$ deactivate
```

## Download the meteorology data and extract wind speed and pressure
