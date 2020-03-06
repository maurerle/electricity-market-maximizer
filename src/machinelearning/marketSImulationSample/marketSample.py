import numpy as np
import matplotlib.pyplot as plt

x_off = np.arange(0, 5)
x = np.arange(0, 7)
off = [0, 1.2, 5, 10, 17]
dem = [17, 10, 5, 3, 0.5, 0.05, 0]

plt.figure()

plt.plot(x_off, off, label='Original Supply', color='C0')
plt.plot(x, dem, label='Original Demand', color='C2')

plt.scatter(2, 5, color='k', s=10)

plt.axhline(y=5, color='k', linestyle='--', linewidth=1)

# 1s Offer Accepted
plt.scatter(1, 1.2, color='k', s=60, marker='*')
plt.scatter(1, 10, color='k', s=10, marker='o')


plt.hlines(y=1.2, xmin=0, xmax=1, color='k', linestyle=':', linewidth=1)
plt.hlines(y=10, xmin=0, xmax=1, color='k', linestyle=':', linewidth=1)
plt.vlines(x=1, ymin=0, ymax=10, color='k', linestyle=':', linewidth=1)

# 1s Offer Rejected
plt.scatter(4, 17, color='k', s=60, marker='x')
plt.scatter(4, 0.5, color='k', s=10, marker='o')


plt.hlines(y=0.5, xmin=0, xmax=4, color='k', linestyle=':', linewidth=1)
plt.hlines(y=17, xmin=0, xmax=4, color='k', linestyle=':', linewidth=1)
plt.vlines(x=4, ymin=0, ymax=17, color='k', linestyle=':', linewidth=1)


plt.yticks([0.5, 1, 5, 10, 17], labels=['$P_{d,2}$', '$P_{o,1}$', '$CP_1$', '$P_{d,1}$', '$P_{o,2}$'])
plt.xticks([1, 4], labels=['$Q_{o,1}$', '$Q_{o,2}$'])



plt.xlabel('Quantity')
plt.ylabel('Price')

plt.legend()
plt.show()