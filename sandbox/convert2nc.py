import os
import sys

from cswot_adcp import data_loader as loader

output_dir = "./nc/"

for file in sys.argv[1:]:

    ds = loader.read_data(file)
    output_file = os.path.join(output_dir, file.split("/")[-1].replace(".STA", ".nc"))
    ds.to_netcdf(output_file)
    print(output_file)

