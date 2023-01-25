import pandas as pd

out_temp = pd.read_csv('./data/out_temp.csv', index_col=[0])
q_hvac = pd.read_csv('./data/qhvac_lump.csv', index_col=[0])/1000
q_int = pd.read_csv('./data/qint_lump.csv', index_col=[0])/1000
q_rad_in = pd.read_csv('./data/qradin_lump.csv', index_col=[0])/1000
q_sol_out = pd.read_csv('./data/qsolout_lump.csv', index_col=[0])/1000
q_win = pd.read_csv('./data/qwin_lump.csv', index_col=[0])/1000
zone_temp = pd.read_csv('./data/zone_temp.csv', index_col=[0])

data = pd.concat([out_temp, q_int, q_hvac, q_win, q_rad_in, zone_temp], axis=1) 
#names=['out_temp', 'q_int', 'q_hvac','q_win', 'q_rad_in', 'zone_temp'])

print(data.head())

data.to_csv('./data/disturbance_1min.csv')

