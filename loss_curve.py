# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-darkgrid')
# Data
df = pd.read_csv('/home/khawar/Downloads/thesis_Graphs/vit_60epochs_loss.csv')
df_Data = df.rolling(100).mean()

df = pd.DataFrame(
    {'x_values': df_Data['Step'],
     'y1_values': df_Data['VIT_60Epoch - loss'],
     'y2_values': df_Data['ViT+OverLP_60_Epochs - loss']
     #'y3_values': df_Data['VIT+OverLP (Ours) - loss'],
     #'y4_values': df_Data['VIT - loss']
     })

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 9,
        }

plt.plot('x_values', 'y1_values', data=df, color='green', linewidth=1.5, label="ViT+Non-OverLP")
plt.plot('x_values', 'y2_values', data=df, color='red', linewidth=1.5, label="ViT+OverLP")

#plt.plot('x_values', 'y2_values', data=df, color='blue', linewidth=1.5, label="DeepViT")
#plt.plot('x_values', 'y1_values', data=df, color='red', linewidth=1.5, label="CaiT")


# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.xticks(weight='bold')
# plt.yticks(weight='bold')

# Display y axis values
#ax = plt.gca()
#ax.set_ylim([0.0, 10000])

plt.xlabel('Number of Iteration', fontdict=font)
plt.ylabel('Training Loss', fontdict=font)
# plt.ylabel('Average accuracy (%)', fontdict=font)

plt.savefig('/media/khawar/HDD_Khawar/Thesis/deeplearning_acc.png')
plt.legend()

# show graph
plt.show()