import scipy.io
import numpy as np

# 读取 .mat 文件
mat = scipy.io.loadmat('/data3/home/yan1/test/BNGNN/Data/sub_002_S_0413.mat')

# 获取数据
data = mat['data']

# 保存为 .npy 文件
np.save('sub_002_S_0413.npy', data)
