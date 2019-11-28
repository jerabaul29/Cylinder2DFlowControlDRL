import numpy as np
import matplotlib.pyplot as plt

if True:
    path = 'full_sym_base/saved_models/debug.csv'
    data = np.genfromtxt(path, delimiter=';')
    
    col = 3

    drag = data[:, col].T
    # Scale
    if col == 3:
        drag = 2*drag/(2*0.05)**2
    else:
        drag = -2*drag/0.05  # Now coefficient

    dt = 1E-4
    t = np.arange(len(drag))*dt

    # For 
    np.savetxt('sym_drag_data', np.c_[t, drag])

t, drag = np.loadtxt('sym_drag_data').T

plt.figure()
plt.plot(t[2:], drag[2:])
plt.show()

# Define stationary state as that where relative difference of conseq steps
# is small
tol = 8E-6
# First for rel error of tol
diff = np.abs(np.diff(drag)/drag[1:])
idx = np.argwhere(diff < tol)[0][0]
# Never wakes up ?
#assert np.all(diff[idx:] < tol)

# Mean of stationalry
print np.mean(drag[idx:]), np.std(drag[idx:])
# 2.93267866609 0.000412274090994
