# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-darkgrid')
# Data
df = pd.read_csv('/home/khawar/Downloads/wandb_export_2021-11-22T15_34_34.633+09_00.csv')
df_Data = df.rolling(100).mean()

df = pd.DataFrame({'x_values': df_Data['Step'],
                   'y1_values': df_Data['OvLP+DW - acc'],
                   'y2_values': df_Data['PVT_Baseline – acc']
                   #'y3_values': df_Data['VIT+OverLP (Ours) - acc'],
                   #'y4_values': df_Data['VIT - acc'],
                   #'y5_values': df_Data['PFRVT(Ours) - acc'],
                   #'y6_values': df_Data['PVT+OverLP+DW+OA+ARcMargin(Ours) - acc'],
                   #'y7_values': df_Data['PVT+OvLP(Ours)+ConvFF(Ours)+FC(Ours) - acc'],
                   #'y8_values': df_Data['PVT+ConVTEM(Ours)+ConFF(Ours) - acc'],
                   #'y9_values': df_Data['PVT+ConVTEM(Ours) - acc'],
                   #'y10_values': df_Data['PFRVT(Ours)+SRA - acc']
                   })

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 9,
        }

# multiple line plots

#plt.plot('x_values', 'y4_values', data=df, color='green', linewidth=1.5, label="VIT")
#plt.plot('x_values', 'y3_values', data=df, color='green', linewidth=1.5, label="+OverLP")
#plt.plot('x_values', 'y9_values', data=df, color='black', linewidth=1.5, label="+ConVTEM")
#plt.plot('x_values', 'y10_values', data=df, color='red', linewidth=1.5, label="+SRA")
#plt.plot('x_values', 'y8_values', data=df, color='green', linewidth=1.5, label="+ConVTEM+ConFF")
#plt.plot('x_values', 'y7_values', data=df, color='blue', linewidth=1.5, label="+OvLP+ConvFF+FC")
#plt.plot('x_values', 'y6_values', data=df, color='red', linewidth=1.5, label="+OverLP+DW+OA+ARcMargin")
#plt.plot('x_values', 'y5_values', data=df, color='black', linewidth=1.5, label="PFRVT")

plt.plot('x_values', 'y1_values', data=df, color='green', linewidth=1.5, label="Convolutional Feed Forward")
plt.plot('x_values', 'y2_values', data=df, color='red', linewidth=1.5, label="Standard Feed Forward")


# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.xticks(weight='bold')
# plt.yticks(weight='bold')

# Display y axis values
# ax = plt.gca()
# ax.set_ylim([0.0, 10000])

plt.xlabel('Number of Iteration', fontdict=font)
# plt.ylabel('Training Loss', fontdict=font)
plt.ylabel('Average accuracy (%)', fontdict=font)

plt.savefig('/media/khawar/HDD_Khawar/Thesis/deeplearning_acc.png')

plt.legend()

# show graph
plt.show()
