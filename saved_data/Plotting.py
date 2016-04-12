from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

y1=[2, 3, 7./3, 3, 7./3, 8./3, 2, 8./3, 8./3, 2, 8./3, 8./3, 8./3, 3, 3, 8./3, 2, 8./3]
y2=[0, 0, 0, 1./3, 0, 0, 0, 0, 0, 1./3, 1./3, 0, 0, 0, 1./3, 1./3, 1./3, 2./3]
for el_index in range(len(y1)):
    y1[el_index] = 100 * y1[el_index]/3.
    y2[el_index] = 100 * y2[el_index]/3.

x1=[]
for i in range(18):
	x1.append(0.32 + 0.02 * i)
print len(x1), len(y1)
data_x = x1
data_y = y1

slope, intercept, r_value, p_value, std_err = stats.linregress(data_x, data_y)
reg_x = np.copy(data_x)
reg_y = []
for i in range(len(data_x)):
    reg_y.append(intercept + data_x[i] * slope)

print "std_err:", std_err

plt.figure(1)
plt.subplot(211)
plt.plot(data_x, data_y, 'bo')
# plt.plot(reg_x, reg_y, 'r')

plt.ylabel('avg. recall rate (%)')
plt.xlabel('neuronal turnover rate')
# plt.text(0.05, 2, r'$\sigma='+"{:1.3f}".format(std_err)+'$')
# plt.text(0.05, 1.85, r'$p='+"{:1.3f}".format(p_value)+'$')
# plt.annotate('linear regression:\nf(x)=' + "{:1.2f}".format(slope)+'x+'+"{:1.2f}".format(intercept),
#              xy=(0.26, 0.26*slope+intercept),
#              xytext=(0.03, 2.25), arrowprops=dict(facecolor='black', shrink=0.05),)

plt.subplot(212)
plt.plot(data_x, y2, 'ro')
plt.ylabel('avg. spurious recall rate (%)')
plt.xlabel('neuronal turnover rate')

plt.show()