from datetime import date, timedelta
import wget
from dotenv import load_dotenv
import os, sys
import subprocess

load_dotenv(override=True)

PREFIX = "Z__C_RJTD_"
SUFFIX = "0000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
PRE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/"
YEAR = 2020

# eg. 
# http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original
#   /2020/01/01/
#   Z__C_RJTD_20200101210000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin

START = date(YEAR, 1, 1)
STOP = date(YEAR+1, 1, 1) # Don't download huge data at once not to place burden on the server.
SAVE_DIR = os.getenv("GRIBFILE_SAVE_DIR") + str(YEAR) + "/"
MAIN_DATA_DIR = os.getenv("DATA_DIR") + str(YEAR) + "/"

os.chdir(SAVE_DIR)

def date_range(start, stop, step=timedelta(days=1)):
    current = start
    while current < stop:
        yield current
        current += step

def download_gribfiles(d, h, condition):
    """
    Download grib files in GRIBFILE_SAVE_DIR, 
    extracting from it and only saving data
    that match the condition.

    Args:
        d (date) : the date of grib data you want to get.
        h (int) : the hour of grib data you want to get.
        condition (str) : the condition of grib data you want to get.
    """
    time_str = f"{d:%Y%m%d}" + str(h).zfill(2)
    filename = PREFIX + time_str + SUFFIX
    url = PRE_URL + f"{d:%Y}/{d:%m}/{d:%d}/" + filename
    wget.download(url)

    result = subprocess.Popen(
        ["wgrib2", filename, "-match", condition],
        stdout=subprocess.PIPE
    )
    out = result.communicate()[0].decode('utf-8').splitlines()
    submsg_count = len(out)

    subprocess.run(
        ["wgrib2", filename, "-for_n", "1:"+str(submsg_count),\
            "-grib", time_str + condition + '.grib2']
    )

    os.remove(filename)

def extract_main_data(d, h, condition, datatypes):
    """
    Extracting main data from grib files in GRIBFILE_SAVE_DIR
    and saving in OUTPUT_DIR. The datatypes of main data are 
    specified by args.

    Args:
        d (date) : the date of grib data you want to extract.
        h (int) : the hour of grib data you want to extract.
        condition (str) : the suffix (condition) of data filename
            you want to extract from.
        datatypes (list) : the datatypes of grib data you want to extract.
    """
    time_str = f"{d:%Y%m%d}" + str(h).zfill(2)
    filename = time_str + condition + '.grib2'

    ids = []
    for datatype in datatypes:
        result = subprocess.Popen(
            ["wgrib2", filename, "-match", datatype],
            stdout=subprocess.PIPE
        )
        out = result.communicate()[0].decode('utf-8')
        id = out.split(':')[0]
        ids.append(id)

    ids_str = '|'.join(ids)
    save_path = MAIN_DATA_DIR + time_str + ".grib2"

    subprocess.run(
        ["wgrib2", filename, "-match", "^(" + ids_str + "):", "-grib", save_path]
    )


def main():
    datatypes = ["PRMSL", "UGRD", "VGRD"]

    for d in date_range(START, STOP):
        for h in range(0, 24, 3):
            download_gribfiles(d, h, "anl")
            extract_main_data(d, h, "anl", datatypes)


if __name__ == '__main__':
    main()
