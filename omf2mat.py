"""
OMF to MAT Converter - Customized for micromagnetic post-processing
Originally based on OOMMFDecode by Mark Mascaro (2010)

This version customized and extended by [Rasool Efhami sisi], [2025]
- Added: conversion to MATLAB-compatible .mat files
- Added: integration with CSV-based energy metadata
- Improved: data reshaping and handling for batch processing

License: GNU General Public License v2 or later (GPLv2+)
Original license preserved as required.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""

import numpy as np # type: ignore
import scipy.io as spio # type: ignore
import os, struct
import pandas as pd # type: ignore

########
# DECS #
########

def _binaryDecode(filehandle, chunksize, decoder, targetarray, headers, extraCaptures):
    valm = headers.get("valuemultiplier",1)
    for k in range(int(headers["znodes"])):
        for j in range(int(headers["ynodes"])):
            for i in range(int(headers["xnodes"])):
                for coord in range(3): #Slowly populate, coordinate by coordinate
                    targetarray[i,j,k,coord] = decoder.unpack(filehandle.read(chunksize))[0] * valm
    print("Decode complete.")
    return (targetarray, headers, extraCaptures)

def _textDecode(filehandle, targetarray, headers, extraCaptures):
    valm = headers.get("valuemultiplier",1)
    for k in range(int(headers["znodes"])):
        for j in range(int(headers["ynodes"])):
            for i in range(int(headers["xnodes"])):
                #numpy is fantastic - splice in a tuple
                text = filehandle.readline().strip().split()
                targetarray[i,j,k] = (float(text[0])*valm, float(text[1])*valm, float(text[2])*valm)
    print("Decode complete.")
    return (targetarray, headers, extraCaptures)


def unpackFile(filename):
    with open(filename, 'rb') as f:
        headers = {} #I know valuemultiplier isn't always present. This is checked later.
        extraCaptures = {'SimTime':-1, 'Iteration':-1, 'Stage':-1, "MIFSource":""}
        #Parse headers
        a = ""
        while not "Begin: Data" in a:
            a = f.readline().decode('utf-8').strip()
            #Determine if it's actually something we need as header data
            for key in ["xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xnodes", "ynodes", "znodes", "valuemultiplier"]:
                if key in a:
                    headers[key] = float(a.split()[2]) #Known position FTW
                #All right, it may also be time data, which we should capture
                if "Total simulation time" in a:
                    #Split on the colon to get the time with units; strip spaces and split on the space to separate time and units
                    #Finally, pluck out the time, stripping defensively (which should be unnecessary).
                    extraCaptures['SimTime'] = float(a.split(":")[-1].strip().split()[0].strip())
                if "Iteration:" in a:
                    #Another tricky split...
                    extraCaptures['Iteration'] = float(a.split(":")[2].split(",")[0].strip())
                if "Stage:" in a:
                    extraCaptures['Stage'] = float(a.split(":")[2].split(",")[0].strip())
                if "MIF source file" in a:
                    extraCaptures['MIFSource'] = a.split(":",2)[2].strip()


        #Initialize array to be populated
        outArray = np.zeros((int(headers["xnodes"]),
                             int(headers["ynodes"]),
                             int(headers["znodes"]),
                             3))

        #Determine decoding mode and use that to populate the array
        print("Data indicator:", a)
        decode = a.split()
        if decode[3] == "Text":
            return _textDecode(f, outArray, headers, extraCaptures)
        elif decode[3] == "Binary" and decode[4] == "4":
            #Determine endianness
            endianflag = f.read(4)
            if struct.unpack(">f", endianflag)[0] == 1234567.0:
                print("Big-endian 4-byte detected.")
                dc = struct.Struct(">f")
            elif struct.unpack("<f", endianflag)[0] == 1234567.0:
                print("Little-endian 4-byte detected.")
                dc = struct.Struct("<f")
            else:
                raise Exception("Can't decode 4-byte byte order mark: " + hex(endianflag))
            return _binaryDecode(f, 4, dc, outArray, headers, extraCaptures)
        elif decode[3] == "Binary" and decode[4] == "8":
            #Determine endianness
            endianflag = f.read(8)
            if struct.unpack(">d", endianflag)[0] == 123456789012345.0:
                print("Big-endian 8-byte detected.")
                dc = struct.Struct(">d")
            elif struct.unpack("<d", endianflag)[0] == 123456789012345.0:
                print("Little-endian 8-byte detected.")
                dc = struct.Struct("<d")
            else:
                raise Exception("Can't decode 8-byte byte order mark: " + hex(endianflag))
            return _binaryDecode(f, 8, dc, outArray, headers, extraCaptures)
        else: 
            raise Exception("Unknown OOMMF data format:" + decode[3] + " " + decode[4])

def convert_omf_to_mat(input_file, output_dir, energy_metadata):
    # Load data from the .omf file
    data, headers, meta = unpackFile(input_file)

    # Extract information from the header
    xnodes = int(headers["xnodes"])
    ynodes = int(headers["ynodes"])
    znodes = int(headers["znodes"])
    xstep = headers["xstepsize"]
    ystep = headers["ystepsize"]
    zstep = headers["zstepsize"]
    xbase = headers.get("xbase", 0.0)
    ybase = headers.get("ybase", 0.0)
    zbase = headers.get("zbase", 0.0)

    # Create output matrix: shape = (xnodes, ynodes, znodes, 6)
    # 6 = [x, y, z, mx, my, mz]
    full_output = np.empty((xnodes, ynodes, znodes, 6), dtype=np.float32)

    # Fill the array
    for k in range(znodes):
        for j in range(ynodes):
            for i in range(xnodes):
                x = xbase + i * xstep
                y = ybase + j * ystep
                z = zbase + k * zstep
                mx, my, mz = data[i, j, k]
                full_output[i, j, k, :] = [x, y, z, mx, my, mz]

    energy_struct = {key: energy_metadata[key] for key in energy_metadata.index}

    # Generate output file path
    file_name = os.path.basename(input_file)
    base_name = os.path.splitext(file_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}.mat")

    # Save to .mat file
    spio.savemat(save_path, {
        'mag_coord_data': full_output,
        'metadata': meta,
        'energy_terms': energy_struct
    })

    print(f"Saved 3D matrix with position and magnetization to: {save_path}")


# Folder path containing .omf files
input_folder = r'C:\Users\Rasool\Desktop\micromag\last\bz_hysteresis_data_70per\drive-1'
csv_path = r"C:\Users\Rasool\Desktop\micromag\last\bz_hysteresis_data_70per.csv"

# Output folder path to save .mat files
output_folder = r"C:\Users\Rasool\Desktop\output_omf2mat\bz_hysteresis_data_70per"

# Loop through all .omf files in the input folder limited by csv files
energy_df = pd.read_csv(csv_path)

num_rows = len(energy_df)
omf_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".omf")])

for idx in range(min(num_rows, len(omf_files))):
    omf_file = omf_files[idx]
    input_file = os.path.join(input_folder, omf_file)
    energy_row = energy_df.iloc[idx]
    convert_omf_to_mat(input_file, output_folder, energy_row)

# for filename in os.listdir(input_folder):
#     if filename.endswith(".omf"):
#         input_file = os.path.join(input_folder, filename)
#         convert_omf_to_mat(input_file, output_folder)