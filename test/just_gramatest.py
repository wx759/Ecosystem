# 去除小于20天

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#导入库
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#为了让图标中显示中文标注加入这两行代码

data = pd.read_csv("est.csv") # 导入csv文件
print(data)
plt.figure(figsize=(10,5))
# 创建图表的尺寸
plt.plot(r.index,r["PM2_5"],color='green',label="每百回合数")
# 北京时间为横轴，PM2.5浓度为纵轴，线条颜色为绿色
plt.title('PM2_5的分析图表')
# 图表的标题
plt.xlabel('时间')
# 图表x轴的标注
plt.ylabel('PM2_5的浓度')
# 图表y轴的标注
plt.legend()
plt.show()
