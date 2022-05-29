import matplotlib.pyplot as plt  # สำหรับ plot graph
import numpy as np  # numpy
from sklearn.linear_model import LinearRegression  # สมการเชิงเส้น


x_data = [1.0, 1.8, 3.0, 4.1, 5.2, 6.0]
y_data = [1, 1.3, 2.2, 2.5, 2.8, 3.6]

x = np.array(x_data)  # เปลี่ยนข้อมูลเป็น numpy array
y = np.array(y_data)

print(x)
print(y)

plt.scatter(x, y)
plt.grid()
plt.show()


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
print("*" * 5)
print(x)
print(y)

model = LinearRegression()  # เริ่มสร้าง model
model.fit(x, y)  # สอนข้อมูล x,y

predict = model.predict([[2.5]]) # คำนวนหาผลลัพ 2.5
print("*" * 5)
print(predict)