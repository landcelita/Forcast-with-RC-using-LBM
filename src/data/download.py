from datetime import date, timedelta
import wget
from dotenv import load_dotenv
import os, sys
import subprocess

load_dotenv(override=True)

def date_range(start, stop, step=timedelta(days=1)):
    current = start
    while current < stop:
        yield current
        current += step

def main():
    PREFIX = "Z__C_RJTD_"
    SUFFIX = "0000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
    PRE_URL = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/"

    # eg. 
    # http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original
    #   /2020/01/01/
    #   Z__C_RJTD_20200101210000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin

    START = date(2020, 9, 9)
    STOP = date(2021, 1, 1) # Don't download huge data at once not to place burden on the server.
    OUTPUT_DIR = os.getenv("OUTPUT_DIR") + "2020"

    DATATYPES = ["PRES", "UGRD", "VGRD"]

    os.chdir(OUTPUT_DIR)

    for d in date_range(START, STOP):
        for h in range(0, 24, 3):
            time_str = f"{d:%Y%m%d}" + str(h).zfill(2)
            filename = PREFIX + time_str + SUFFIX
            url = PRE_URL + f"{d:%Y}/{d:%m}/{d:%d}/" + filename
            wget.download(url)

            for datatype in DATATYPES:
                result = subprocess.Popen(["wgrib2", filename, "-match", datatype, "-match", "anl"], \
                            stdout=subprocess.PIPE)
                out = result.communicate()[0].decode('utf-8')
                id = out.split(':')[0]
                subprocess.run(["wgrib2", filename, '-d', id, '-grib', time_str + datatype + '.grib2'])

            grib_files = [time_str + datatype + '.grib2' for datatype in DATATYPES]
            merged_filepath = time_str + "_merged.grib2"
            
            with open(merged_filepath, 'wb') as saveFile:
                for f in grib_files:
                    data = open(f, "rb").read()
                    saveFile.write(data)
                    saveFile.flush()

            for datatype in DATATYPES:
                os.remove(time_str + datatype + '.grib2')
            os.remove(filename)





            
            
        
            

if __name__ == '__main__':
    main()
