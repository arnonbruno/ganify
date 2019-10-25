from model.model import Ganify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_parquet(r'D:\Arquivos\Projetos\Datasets\Risk\risk.parquet')
data = data[(data['1h_porte_small'] == 1) & (
    data['flag_sinistro_indenizados'] == 1)]
features = data.drop('flag_sinistro_indenizados', axis=1)
target = data['flag_sinistro_indenizados']

ganify = Ganify()
ganify.fit_data(
    x_train=features, y_train=target, epochs=1, cols_names=data.columns)

a = ganify.create_bulk(10, output=1)
a.shape
a
len(ganify.cols_names)

ganify.plot_performance()

a[0]
ganify.scaler
plt.plot(ganify.d1_hist)

print(len(np.unique(target)))
a = np.array(target)
len(np.unique(a))
print('oi')
