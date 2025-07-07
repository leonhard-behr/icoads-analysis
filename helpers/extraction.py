"""loads, extracts and converts all MSG.1 data files in a given folder into a single HuggingFace dataset."""
import tarfile
import os
import tempfile
import gzip
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets

base_dir = r"enter/base/dir/here"

filelist_g3 = [
    'MSG1_R3.0.0_ENH_G3_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G3_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G3_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G3_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G3_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G3_2010-2019.tar',
]

filelist_g4 = [
    'MSG1_R3.0.0_ENH_G4_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G4_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G4_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G4_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G4_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G4_2010-2019.tar',
]

filelist_g5 = [
    'MSG1_R3.0.0_ENH_G5_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G5_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G5_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G5_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G5_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G5_2010-2019.tar',
]

filelist_g6 = [
    'MSG1_R3.0.0_ENH_G6_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G6_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G6_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G6_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G6_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G6_2010-2019.tar',
]

filelist_g7 = [
    'MSG1_R3.0.0_ENH_G7_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G7_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G7_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G7_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G7_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G7_2010-2019.tar',
]

filelist_g9 = [
    'MSG1_R3.0.0_ENH_G9_1960-1969.tar',
    'MSG1_R3.0.0_ENH_G9_1970-1979.tar',
    'MSG1_R3.0.0_ENH_G9_1980-1989.tar',
    'MSG1_R3.0.0_ENH_G9_1990-1999.tar',
    'MSG1_R3.0.0_ENH_G9_2000-2009.tar',
    'MSG1_R3.0.0_ENH_G9_2010-2019.tar',
]

# group variable definitions based on FORTRAN FORMAT strings
GROUP_DEFINITIONS = {
    3: {
        'name': 'Basic Ocean-Atmosphere Variables',
        'variables': {
            'var1': 'sea_surface_temp',     # S - Sea surface temperature
            'var2': 'air_temp',             # A - Air temperature  
            'var3': 'specific_humidity',    # Q - Specific humidity
            'var4': 'rainfall_rate'         # R - Rainfall rate
        },
        'description': 'Basic temperature, humidity, and precipitation measurements'
    },
    4: {
        'name': 'Wind and Pressure Variables',
        'variables': {
            'var1': 'wind_speed',           # W - Wind speed
            'var2': 'wind_u_component',     # U - Wind U-component (eastward)
            'var3': 'wind_v_component',     # V - Wind V-component (northward)
            'var4': 'sea_level_pressure'    # P - Sea level pressure
        },
        'description': 'Wind speed, components, and atmospheric pressure'
    },
    5: {
        'name': 'Cloud and Wind Products',
        'variables': {
            'var1': 'cloud_coverage',       # C - Cloud coverage
            'var2': 'rainfall_amount',      # R - Rainfall amount
            'var3': 'wind_stress_u',        # X = W*U - Wind stress U-component
            'var4': 'wind_stress_v'         # Y = W*V - Wind stress V-component
        },
        'description': 'Cloud, precipitation, and wind stress variables'
    },
    6: {
        'name': 'Temperature and Humidity Differences',
        'variables': {
            'var1': 'sea_air_temp_diff',    # D = S-A - Sea-air temperature difference
            'var2': 'temp_diff_wind',       # E = (S-A)*W - Temperature difference × wind
            'var3': 'humidity_deficit',     # F = QS-Q - Humidity deficit (saturation - actual)
            'var4': 'humidity_flux'         # G = (QS-Q)*W - Humidity flux
        },
        'description': 'Temperature and humidity differences and fluxes'
    },
    7: {
        'name': 'Wind-Temperature/Humidity Products',
        'variables': {
            'var1': 'wind_u_temp',          # I = U*A - U-wind × air temperature
            'var2': 'wind_v_temp',          # J = V*A - V-wind × air temperature
            'var3': 'wind_u_humidity',      # K = U*Q - U-wind × specific humidity
            'var4': 'wind_v_humidity'       # L = V*Q - V-wind × specific humidity
        },
        'description': 'Wind-temperature and wind-humidity interaction products'
    },
    9: {
        'name': 'Moisture Flux and Turbulence',
        'variables': {
            'var1': 'moisture_flux_u',      # M = (QS-Q)*U - Moisture flux U-direction
            'var2': 'moisture_flux_v',      # N = (QS-Q)*V - Moisture flux V-direction
            'var3': 'wind_cubed_b1',        # B1 = W³ - Wind speed cubed (method 1)
            'var4': 'wind_cubed_b2'         # B2 = W³ - Wind speed cubed (method 2)
        },
        'description': 'Advanced moisture flux and turbulence parameters'
    }
}

def unpack_msg1_record(record_bytes):
    """
    unpack MSG.1 binary record based on FORTRAN UNPACKx subroutine
    each record is 64 bytes of packed binary data
    """
    if len(record_bytes) != 64:
        return None
    
    # converting bytes to list of integers (equivalent to ICHAR in FORTRAN)
    chars = list(record_bytes)
    
    # unpacking according to FORTRAN UNPACKx subroutine - exactly as in FORTRAN
    coded = [0] * 50  # 49 + 1 for 0-indexing
    
    coded[1] = chars[2]  # ICHAR(RPT(3:3))
    coded[2] = chars[3] // 16  # ICHAR(RPT(4:4))/16
    coded[3] = (chars[3] % 16) // 2  # MOD(ICHAR(RPT(4:4)),16)/2
    coded[4] = ((chars[3] % 2) * 256 + chars[4]) * 2 + chars[5] // 128
    coded[5] = (chars[5] % 128) * 4 + chars[6] // 64
    coded[6] = (chars[6] % 64) // 8
    coded[7] = chars[6] % 8
    coded[8] = chars[7] // 16
    coded[9] = chars[7] % 16
    
    # 16-bit values - positions 10-33
    coded[10] = chars[8] * 256 + chars[9]
    coded[11] = chars[10] * 256 + chars[11]
    coded[12] = chars[12] * 256 + chars[13]
    coded[13] = chars[14] * 256 + chars[15]
    coded[14] = chars[16] * 256 + chars[17]
    coded[15] = chars[18] * 256 + chars[19]
    coded[16] = chars[20] * 256 + chars[21]
    coded[17] = chars[22] * 256 + chars[23]
    coded[18] = chars[24] * 256 + chars[25]
    coded[19] = chars[26] * 256 + chars[27]
    coded[20] = chars[28] * 256 + chars[29]
    coded[21] = chars[30] * 256 + chars[31]
    coded[22] = chars[32] * 256 + chars[33]
    coded[23] = chars[34] * 256 + chars[35]
    coded[24] = chars[36] * 256 + chars[37]
    coded[25] = chars[38] * 256 + chars[39]
    coded[26] = chars[40] * 256 + chars[41]
    coded[27] = chars[42] * 256 + chars[43]
    coded[28] = chars[44] * 256 + chars[45]
    coded[29] = chars[46] * 256 + chars[47]
    coded[30] = chars[48] * 256 + chars[49]
    coded[31] = chars[50] * 256 + chars[51]
    coded[32] = chars[52] * 256 + chars[53]
    coded[33] = chars[54] * 256 + chars[55]
    
    # 4-bit nibbles - positions 34-49
    coded[34] = chars[56] // 16
    coded[35] = chars[56] % 16
    coded[36] = chars[57] // 16
    coded[37] = chars[57] % 16
    coded[38] = chars[58] // 16
    coded[39] = chars[58] % 16
    coded[40] = chars[59] // 16
    coded[41] = chars[59] % 16
    coded[42] = chars[60] // 16
    coded[43] = chars[60] % 16
    coded[44] = chars[61] // 16
    coded[45] = chars[61] % 16
    coded[46] = chars[62] // 16
    coded[47] = chars[62] % 16
    coded[48] = chars[63] // 16
    coded[49] = chars[63] % 16
    
    return coded

def get_scaling_factors(group):
    """
    get base values and unit scaling factors for each group based on FORTRAN BLOCK DATA - these would need to be determined from the complete FORTRAN source for all groups
    """    
    if group == 3:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -501., -8801., -1., -1.,        # positions 10-13 (S1)
                 -501., -8801., -1., -1.,        # positions 14-17 (S3)
                 -501., -8801., -1., -1.,        # positions 18-21 (S5)
                 -501., -8801., -1., -1.,        # positions 22-25 (M)
                 0., 0., 0., 0.,                 # positions 26-29 (N)
                 -1., -1., -1., -1.,             # positions 30-33 (S)
                 0., 0., 0., 0.,                 # positions 34-37 (D)
                 -1., -1., -1., -1.,             # positions 38-41 (H)
                 -1., -1., -1., -1.,             # positions 42-45 (X)
                 -1., -1., -1., -1.]             # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.01, 0.01, 0.01, 0.1,        # positions 10-13 (S1): S, A, Q, R
                  0.01, 0.01, 0.01, 0.1,        # positions 14-17 (S3): S, A, Q, R
                  0.01, 0.01, 0.01, 0.1,        # positions 18-21 (S5): S, A, Q, R
                  0.01, 0.01, 0.01, 0.1,        # positions 22-25 (M):  S, A, Q, R
                  1., 1., 1., 1.,                # positions 26-29 (N)
                  0.01, 0.01, 0.01, 0.1,        # positions 30-33 (S)
                  2., 2., 2., 2.,                # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,           # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,           # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]           # positions 46-49 (Y)
    
    elif group == 4:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -1., -10221., -10221., 86999.,  # positions 10-13 (S1)
                 -1., -10221., -10221., 86999.,  # positions 14-17 (S3)
                 -1., -10221., -10221., 86999.,  # positions 18-21 (S5)
                 -1., -10221., -10221., 86999.,  # positions 22-25 (M)
                 0., 0., 0., 0.,                 # positions 26-29 (N)
                 -1., -1., -1., -1.,             # positions 30-33 (S)
                 0., 0., 0., 0.,                 # positions 34-37 (D)
                 -1., -1., -1., -1.,             # positions 38-41 (H)
                 -1., -1., -1., -1.,             # positions 42-45 (X)
                 -1., -1., -1., -1.]             # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.01, 0.01, 0.01, 0.01,       # positions 10-13 (S1): W, U, V, P
                  0.01, 0.01, 0.01, 0.01,       # positions 14-17 (S3): W, U, V, P
                  0.01, 0.01, 0.01, 0.01,       # positions 18-21 (S5): W, U, V, P
                  0.01, 0.01, 0.01, 0.01,       # positions 22-25 (M):  W, U, V, P
                  1., 1., 1., 1.,                # positions 26-29 (N)
                  0.01, 0.01, 0.01, 0.01,       # positions 30-33 (S)
                  2., 2., 2., 2.,                # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,           # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,           # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]           # positions 46-49 (Y)
    
    elif group == 5:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -1., -1., -30001., -30001.,     # positions 10-13 (S1)
                 -1., -1., -30001., -30001.,     # positions 14-17 (S3)
                 -1., -1., -30001., -30001.,     # positions 18-21 (S5)
                 -1., -1., -30001., -30001.,     # positions 22-25 (M)
                 0., 0., 0., 0.,                 # positions 26-29 (N)
                 -1., -1., -1., -1.,             # positions 30-33 (S)
                 0., 0., 0., 0.,                 # positions 34-37 (D)
                 -1., -1., -1., -1.,             # positions 38-41 (H)
                 -1., -1., -1., -1.,             # positions 42-45 (X)
                 -1., -1., -1., -1.]             # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.1, 0.1, 0.1, 0.1,           # positions 10-13 (S1): C, R, X=W*U, Y=W*V
                  0.1, 0.1, 0.1, 0.1,           # positions 14-17 (S3): C, R, X=W*U, Y=W*V
                  0.1, 0.1, 0.1, 0.1,           # positions 18-21 (S5): C, R, X=W*U, Y=W*V
                  0.1, 0.1, 0.1, 0.1,           # positions 22-25 (M):  C, R, X=W*U, Y=W*V
                  1., 1., 1., 1.,                # positions 26-29 (N)
                  0.1, 0.1, 0.1, 0.1,           # positions 30-33 (S)
                  2., 2., 2., 2.,                # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,           # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,           # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]           # positions 46-49 (Y)
    
    elif group == 6:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -6301., -10001., -4001., -10001.,  # positions 10-13 (S1)
                 -6301., -10001., -4001., -10001.,  # positions 14-17 (S3)
                 -6301., -10001., -4001., -10001.,  # positions 18-21 (S5)
                 -6301., -10001., -4001., -10001.,  # positions 22-25 (M)
                 0., 0., 0., 0.,                    # positions 26-29 (N)
                 -1., -1., -1., -1.,                # positions 30-33 (S)
                 0., 0., 0., 0.,                    # positions 34-37 (D)
                 -1., -1., -1., -1.,                # positions 38-41 (H)
                 -1., -1., -1., -1.,                # positions 42-45 (X)
                 -1., -1., -1., -1.]                # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.01, 0.1, 0.01, 0.1,          # positions 10-13 (S1): D=S-A, E=(S-A)*W, F=QS-Q, G=(QS-Q)*W
                  0.01, 0.1, 0.01, 0.1,          # positions 14-17 (S3): D=S-A, E=(S-A)*W, F=QS-Q, G=(QS-Q)*W
                  0.01, 0.1, 0.01, 0.1,          # positions 18-21 (S5): D=S-A, E=(S-A)*W, F=QS-Q, G=(QS-Q)*W
                  0.01, 0.1, 0.01, 0.1,          # positions 22-25 (M):  D=S-A, E=(S-A)*W, F=QS-Q, G=(QS-Q)*W
                  1., 1., 1., 1.,                 # positions 26-29 (N)
                  0.01, 0.1, 0.01, 0.1,          # positions 30-33 (S)
                  2., 2., 2., 2.,                 # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,            # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,            # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]            # positions 46-49 (Y)
    
    elif group == 7:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -20001., -20001., -10001., -10001.,  # positions 10-13 (S1)
                 -20001., -20001., -10001., -10001.,  # positions 14-17 (S3)
                 -20001., -20001., -10001., -10001.,  # positions 18-21 (S5)
                 -20001., -20001., -10001., -10001.,  # positions 22-25 (M)
                 0., 0., 0., 0.,                      # positions 26-29 (N)
                 -1., -1., -1., -1.,                  # positions 30-33 (S)
                 0., 0., 0., 0.,                      # positions 34-37 (D)
                 -1., -1., -1., -1.,                  # positions 38-41 (H)
                 -1., -1., -1., -1.,                  # positions 42-45 (X)
                 -1., -1., -1., -1.]                  # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.1, 0.1, 0.1, 0.1,            # positions 10-13 (S1): I=U*A, J=V*A, K=U*Q, L=V*Q
                  0.1, 0.1, 0.1, 0.1,            # positions 14-17 (S3): I=U*A, J=V*A, K=U*Q, L=V*Q
                  0.1, 0.1, 0.1, 0.1,            # positions 18-21 (S5): I=U*A, J=V*A, K=U*Q, L=V*Q
                  0.1, 0.1, 0.1, 0.1,            # positions 22-25 (M):  I=U*A, J=V*A, K=U*Q, L=V*Q
                  1., 1., 1., 1.,                 # positions 26-29 (N)
                  0.1, 0.1, 0.1, 0.1,            # positions 30-33 (S)
                  2., 2., 2., 2.,                 # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,            # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,            # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]            # positions 46-49 (Y)

    elif group == 9:
        fbase = [0, 1799., 0., -1., -1., -181., -1., -1., 0., 0.,
                 -10001., -10001., -1., -1.,      # positions 10-13 (S1)
                 -10001., -10001., -1., -1.,      # positions 14-17 (S3)
                 -10001., -10001., -1., -1.,      # positions 18-21 (S5)
                 -10001., -10001., -1., -1.,      # positions 22-25 (M)
                 0., 0., 0., 0.,                  # positions 26-29 (N)
                 -1., -1., -1., -1.,              # positions 30-33 (S)
                 0., 0., 0., 0.,                  # positions 34-37 (D)
                 -1., -1., -1., -1.,              # positions 38-41 (H)
                 -1., -1., -1., -1.,              # positions 42-45 (X)
                 -1., -1., -1., -1.]              # positions 46-49 (Y)
        
        funits = [0, 1., 1., 1., 0.5, 0.5, 1., 1., 1., 1.,
                  0.1, 0.1, 0.5, 5.,             # positions 10-13 (S1): M=(QS-Q)*U, N=(QS-Q)*V, B1=W³, B2=W³
                  0.1, 0.1, 0.5, 5.,             # positions 14-17 (S3): M=(QS-Q)*U, N=(QS-Q)*V, B1=W³, B2=W³
                  0.1, 0.1, 0.5, 5.,             # positions 18-21 (S5): M=(QS-Q)*U, N=(QS-Q)*V, B1=W³, B2=W³
                  0.1, 0.1, 0.5, 5.,             # positions 22-25 (M):  M=(QS-Q)*U, N=(QS-Q)*V, B1=W³, B2=W³
                  1., 1., 1., 1.,                 # positions 26-29 (N)
                  0.1, 0.1, 0.5, 5.,             # positions 30-33 (S)
                  2., 2., 2., 2.,                 # positions 34-37 (D)
                  0.1, 0.1, 0.1, 0.1,            # positions 38-41 (H)
                  0.1, 0.1, 0.1, 0.1,            # positions 42-45 (X)
                  0.1, 0.1, 0.1, 0.1]            # positions 46-49 (Y)
    
    else:
        raise ValueError(f"Unknown group: {group}") 
    
    return fbase, funits

def convert_to_true_values(coded, group):
    """
    converts coded values to true physical values using FORTRAN scaling factors
    """
    FMISS = -9999.0
    
    fbase, funits = get_scaling_factors(group)
    
    # converting coded to true values
    ftrue = [0] * 50
    
    # header values (positions 1-9)
    ftrue[1] = (coded[1] + fbase[1]) * funits[1]  # Year
    ftrue[2] = (coded[2] + fbase[2]) * funits[2]  # Month
    ftrue[3] = (coded[3] + fbase[3]) * funits[3]  # Box size
    ftrue[4] = (coded[4] + fbase[4]) * funits[4]  # Longitude
    ftrue[5] = (coded[5] + fbase[5]) * funits[5]  # Latitude
    ftrue[6] = (coded[6] + fbase[6]) * funits[6]  # Platform ID 1
    ftrue[7] = (coded[7] + fbase[7]) * funits[7]  # Platform ID 2
    ftrue[8] = (coded[8] + fbase[8]) * funits[8]  # Group
    ftrue[9] = coded[9]                           # Checksum
    
    # data values (positions 10-49)
    for i in range(10, 50):
        if coded[i] == 0:
            ftrue[i] = FMISS
        else:
            ftrue[i] = (coded[i] + fbase[i]) * funits[i]
    
    return ftrue

def create_record_columns(ftrue, group, source_file):
    """
    creates a record dictionary with appropriate column names for each group
    """
    group_def = GROUP_DEFINITIONS.get(group, GROUP_DEFINITIONS[3])
    var_names = group_def['variables']
    
    record = {
        'year': int(ftrue[1]),
        'month': int(ftrue[2]),
        'longitude': ftrue[4],
        'latitude': ftrue[5],
        'box_size_degrees': ftrue[3],
        'platform_id1': ftrue[6],
        'platform_id2': ftrue[7],
        'data_group': int(ftrue[8]),
        'checksum': ftrue[9],
        'source_file': source_file,
    }
    
    # adding variables with proper names for this group
    for i, (var_key, var_name) in enumerate(var_names.items(), 1):
        record[f'{var_name}_tercile1'] = ftrue[9 + i]     # S1 (positions 10-13)
        record[f'{var_name}_median'] = ftrue[13 + i]      # S3 (positions 14-17)
        record[f'{var_name}_tercile3'] = ftrue[17 + i]    # S5 (positions 18-21)
        record[f'{var_name}_mean'] = ftrue[21 + i]        # M  (positions 22-25)
    
    return record

def parse_tar_file(tar_path):
    """
    Parse a single tar file and return list of records
    """
    print(f"Processing: {os.path.basename(tar_path)}")
    
    # group number extraction form filename
    if '_G3_' in tar_path:
        expected_group = 3
    elif '_G4_' in tar_path:
        expected_group = 4
    elif '_G5_' in tar_path:
        expected_group = 5
    elif '_G6_' in tar_path:
        expected_group = 6
    elif '_G7_' in tar_path:
        expected_group = 7
    elif '_G9_' in tar_path:
        expected_group = 9
    else:
        raise ValueError(f"Unknown group in filename: {tar_path}")
    
    print(f"Expected group: {expected_group} - {GROUP_DEFINITIONS[expected_group]['name']}")
    
    records = []
    
    with tarfile.open(tar_path, 'r') as tar:
        with tempfile.TemporaryDirectory() as temp_dir:
            tar.extractall(temp_dir)
            
            gz_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.gz'):
                        gz_files.append(os.path.join(root, file))
            
            print(f"Found {len(gz_files)} monthly files")
            
            for gz_file in gz_files:
                try:
                    with gzip.open(gz_file, 'rb') as f:
                        data = f.read()
                    
                    file_records = 0
                    for offset in range(0, len(data), 64):
                        if offset + 64 > len(data):
                            break
                            
                        record_bytes = data[offset:offset+64]
                        
                        if len(record_bytes) == 64 and (record_bytes[1] % 16) == 1:
                            coded = unpack_msg1_record(record_bytes)
                            if coded:
                                actual_group = coded[8]
                                
                                ftrue = convert_to_true_values(coded, actual_group)
                                
                                record = create_record_columns(ftrue, actual_group, os.path.basename(gz_file))
                                
                                records.append(record)
                                file_records += 1
                    
                    print(f"{os.path.basename(gz_file)}: {file_records} records")
                    
                except Exception as e:
                    print(f"Error processing {os.path.basename(gz_file)}: {e}")
    
    return records

def validate_and_create_path(path):
    path = os.path.normpath(path)
    
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    print(f"Validated output path: {path}")
    
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Created parent directory: {parent_dir}")
        except Exception as e:
            raise OSError(f"Cannot create parent directory {parent_dir}: {e}")
    
    if not os.access(parent_dir, os.W_OK):
        raise OSError(f"No write permission for directory: {parent_dir}")
    
    if os.path.exists(path):
        if not os.access(path, os.W_OK):
            raise OSError(f"No write permission for existing path: {path}")
        print(f"Warning: Output path already exists and will be overwritten: {path}")
    
    return path


def parse_all_groups_optimized_hf(base_directory, file_list, output_path=None, chunk_size=50000, separate_groups=True):
    """Parse all group files with extensive debugging"""
    print("="*80)
    print("ICOADS MSG.1 MULTI-GROUP PARSER (HUGGINGFACE OPTIMIZED)")
    print("="*80)
    
    if output_path is None:
        output_path = r"C:\Users\leobe\OneDrive\Documents\KIK 4-Thinkpad-Behr\KI-Life-Sciences\notebook\hf_dataset_2\icoads_ENH_G3_hf_dataset"
    
    try:
        output_path = validate_and_create_path(output_path)
    except OSError as e:
        print(f"ERROR: {e}")
        return None, None
    
    print("\nVerifying input files...")
    for filename in file_list:
        tar_path = os.path.join(base_directory, filename)
        if not os.path.exists(tar_path):
            print(f"ERROR: File not found: {tar_path}")
            return None, None
        else:
            print(f"✓ Found: {filename}")
    
    total_records = 0
    chunk_buffer = []
    group_datasets = {} if separate_groups else None
    
    for filename in file_list:
        tar_path = os.path.join(base_directory, filename)
        
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(tar_path)}")
        print(f"{'='*60}")
        
        try:
            records = parse_tar_file(tar_path)
            print(f"✓ Extracted {len(records)} records from {filename}")
            
            if not records:
                print(f"WARNING: No records extracted from {filename}")
                continue
                
        except Exception as e:
            print(f"ERROR extracting from {filename}: {e}")
            continue
        
        chunk_buffer.extend(records)
        total_records += len(records)
        
        print(f"Records from this file: {len(records):,}")
        print(f"Total records so far: {total_records:,}")
        print(f"Buffer size: {len(chunk_buffer):,}")
        
        while len(chunk_buffer) >= chunk_size:
            chunk_records = chunk_buffer[:chunk_size]
            chunk_buffer = chunk_buffer[chunk_size:]
            
            print(f"\n--- Processing chunk of {len(chunk_records)} records ---")
            
            try:
                chunk_df = process_chunk_hf(chunk_records)
                
                if chunk_df.empty:
                    print("ERROR: Chunk DataFrame is empty after processing!")
                    continue
                
                print("Converting DataFrame to HuggingFace Dataset...")
                chunk_dataset = Dataset.from_pandas(chunk_df, preserve_index=False)
                print(f"✓ Created HF Dataset with {len(chunk_dataset)} records")
                
                if separate_groups:
                    unique_groups = chunk_df['data_group'].unique()
                    print(f"Found groups in chunk: {unique_groups}")
                    
                    for group in unique_groups:
                        group_data = chunk_df[chunk_df['data_group'] == group]
                        print(f"Group {group}: {len(group_data)} records")
                        
                        if len(group_data) > 0:
                            group_dataset = Dataset.from_pandas(group_data, preserve_index=False)
                            
                            if group not in group_datasets:
                                group_datasets[group] = []
                            group_datasets[group].append(group_dataset)
                            print(f"✓ Added to Group {group} datasets (now {len(group_datasets[group])} chunks)")
                
                print(f"✓ Processed chunk of {len(chunk_dataset):,} records")
                del chunk_df, chunk_dataset, chunk_records
                
            except Exception as e:
                print(f"ERROR processing chunk: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if chunk_buffer:
        print(f"\n--- Processing final chunk of {len(chunk_buffer)} records ---")
        try:
            chunk_df = process_chunk_hf(chunk_buffer)
            
            if not chunk_df.empty:
                final_dataset = Dataset.from_pandas(chunk_df, preserve_index=False)
                
                if separate_groups:
                    unique_groups = chunk_df['data_group'].unique()
                    for group in unique_groups:
                        group_data = chunk_df[chunk_df['data_group'] == group]
                        if len(group_data) > 0:
                            group_dataset = Dataset.from_pandas(group_data, preserve_index=False)
                            
                            if group not in group_datasets:
                                group_datasets[group] = []
                            group_datasets[group].append(group_dataset)
                
                print(f"✓ Processed final chunk of {len(final_dataset):,} records")
                del chunk_df, final_dataset
        except Exception as e:
            print(f"ERROR processing final chunk: {e}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"Total records processed: {total_records:,}")
    print(f"{'='*60}")
    
    if separate_groups and group_datasets:
        print("\nCombining datasets by groups...")
        combined_datasets = {}
        
        for group, dataset_chunks in group_datasets.items():
            print(f"\nCombining Group {group}:")
            print(f"  Number of chunks: {len(dataset_chunks)}")
            
            try:
                if len(dataset_chunks) == 1:
                    combined_datasets[group] = dataset_chunks[0]
                    print(f"  ✓ Single chunk, using directly")
                else:
                    print(f"  Concatenating {len(dataset_chunks)} chunks...")
                    
                    combined_datasets[group] = concatenate_datasets(dataset_chunks)
                    print(f"  ✓ Concatenated successfully")
                
                print(f"  Final Group {group}: {len(combined_datasets[group]):,} records")
                
            except Exception as e:
                print(f"  ERROR combining Group {group}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not combined_datasets:
            print("ERROR: No datasets were successfully combined!")
            return None, None
        
        print(f"\nSaving {len(combined_datasets)} groups to disk...")
        try:
            string_keyed_datasets = {str(group): dataset for group, dataset in combined_datasets.items()}
            
            print("Creating DatasetDict...")
            dataset_dict = DatasetDict(string_keyed_datasets)
            
            print(f"Saving to: {output_path}")
            dataset_dict.save_to_disk(output_path)
            
            print(f"\n🎉 SUCCESS! Datasets saved!")
            print("Group structure:")
            for group, dataset in combined_datasets.items():
                print(f"  Group {group}: {len(dataset):,} records")
                if group in GROUP_DEFINITIONS:
                    print(f"    Variables: {GROUP_DEFINITIONS[group]['name']}")
            
            return output_path, combined_datasets
            
        except Exception as e:
            print(f"ERROR saving datasets: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    else:
        print("ERROR: No group datasets were created!")
        if group_datasets:
            print(f"Group datasets keys: {list(group_datasets.keys())}")
            for group, chunks in group_datasets.items():
                print(f"  Group {group}: {len(chunks)} chunks")
        return None, None


def process_chunk_hf(records):    
    if not records:
        print("WARNING: Empty records passed to process_chunk_hf")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    print(f"DataFrame created with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if df.empty:
        print("ERROR: DataFrame is empty after creation!")
        return df
    
    df = df.replace(-9999.0, None)
    
    essential_cols = ['year', 'month', 'data_group']
    for col in essential_cols:
        if col in df.columns:
            none_count = df[col].isna().sum()
            print(f"Column {col}: {none_count} None values out of {len(df)}")
        else:
            print(f"ERROR: Essential column {col} missing!")
    
    original_len = len(df)
    for col in essential_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    print(f"After removing invalid rows: {len(df)} (removed {original_len - len(df)})")
    
    if len(df) == 0:
        print("ERROR: All rows removed due to missing essential data!")
        return df
    
    df['date_string'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    for col in ['year', 'month', 'data_group', 'checksum']:
        if col in df.columns:
            df[col] = df[col].astype(int)
            print(f"Converted {col} to int, dtype: {df[col].dtype}")
    
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        df[col] = df[col].astype(float)
    
    for col in ['source_file', 'date_string']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"=== CHUNK PROCESSING COMPLETE ===\n")
    return df


def ensure_json_serializable_types(dataset):
    data_dict = {}
    
    df = dataset.to_pandas()
    columns = df.columns.tolist()
    
    for col in columns:
        data_dict[col] = []
    
    for i in range(len(dataset)):
        row = dataset[i]
        for col in columns:
            value = row[col]            
            if value is None:
                native_value = None
            elif isinstance(value, (np.int8, np.int16, np.int32, np.int64)):
                native_value = int(value)
            elif isinstance(value, (np.float16, np.float32, np.float64)):
                native_value = float(value)
            else:
                native_value = value
            
            data_dict[col].append(native_value)
    
    return Dataset.from_dict(data_dict)

def debug_dataset_types(dataset):
    df = dataset.to_pandas()
    print("Dataset dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
        
        if hasattr(dtype, 'type'):
            print(f"    numpy type: {dtype.type}")
        
        sample_vals = df[col].dropna().head(3)
        for val in sample_vals:
            print(f"    sample value type: {type(val)}")
        print()

if __name__ == "__main__":
    output_path = r"enter/output/path/here"
    
    result_path, datasets = parse_all_groups_optimized_hf(
        base_dir,
        filelist_g3,
        output_path,
        chunk_size=25000,
        separate_groups=True
    )
    
    if result_path:
        print(f"\nHuggingFace dataset successfully created: {result_path}")
    else:
        print("HuggingFace dataset creation failed!")