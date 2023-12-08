import math

A1 = 0.104313958
E1 = -0.214534747
A2 = 0.189389672
E2 = -0.326864702

x1 = -1714195.10618333
y1 = 3581986.37675711
z1 = 4982889.52709341
x2 = -1714194.88339328
y2 = 3581988.94015599
z2 = 4982887.74859955


# ---------------------------------------------------------------------
m1 = math.cos(A1) * (x1 - x2) + math.tan(E1) * (y1 - y2) + math.sin(A1) * (z1 - z2)
m2 = math.cos(A2) * (x2 - x1) + math.tan(E2) * (y2 - y1) + math.sin(A2) * (z2 - z1)
K = (math.cos(A1 - A2) + math.tan(E1) * math.tan(E2)) ** 2 - (1 / (math.cos(E1) * math.cos(E2))) ** 2
L1 = (m2 * (math.cos(A1 - A2) + math.tan(E1) * math.tan(E2)) + m1 * (1 / (math.cos(E2))) ** 2) / K
L2 = (m1 * (math.cos(A1 - A2) + math.tan(E1) * math.tan(E2)) + m2 * (1 / (math.cos(E1))) ** 2) / K

x11 = x1 + L1 * math.cos(A1)
y11 = y1 + L1 * math.tan(E1)
z11 = z1 + L1 * math.sin(A1)

x21 = x2 + L2 * math.cos(A2)
y21 = y2 + L2 * math.tan(E2)
z21 = z2 + L2 * math.sin(A2)

rho = 0.5  # You need to define the value of rho here

xm = rho * x11 + (1 - rho) * x21
ym = rho * y11 + (1 - rho) * y21
zm = rho * z11 + (1 - rho) * z21

print(f"xm: {xm}, ym: {ym}, zm: {zm}")
