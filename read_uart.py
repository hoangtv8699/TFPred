import serial
import time
import os
import pandas as pd
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


# hàm tạo kết nối và đọc dữ liệu từ UART
def readData():
    # ser = serial.Serial(
    #     port='/dev/ttyUSB0',
    #     baudrate=9600,
    # )
    # data = ser.readline()
    data = random.random()
    return data


if __name__ == "__main__":
    plt.ion()
    # đặt tên file là thời gian chạy
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    save_path = dt_string + '.xlsx'

    # vẽ đồ thị
    x_vec = deque(maxlen=50)
    y_vec = deque(maxlen=50)
    graph = False

    data_time = []
    data_all = []
    while True:
        # đọc dữ liệu từ UART
        data = readData()
        # lấy ngày và giờ thời điểm đọc được dữ liệu
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        # lưu dữ liệu và ghi file excel
        data_time.append(dt_string)
        data_all.append(data)
        pd.DataFrame({'time': data_time,
                      'data': data_all}) \
            .to_excel(save_path)

        # chỉ lấy 50 mẫu gần nhất để cập nhật đồ thị
        x_vec.append(now.strftime("%H:%M:%S"))
        y_vec.append(data)

        # cập nhật đồ thị
        plt.clf()
        plt.plot(x_vec, y_vec, '-o', alpha=0.8, color='green')
        plt.ylabel('data')
        plt.xlabel('time')
        plt.ylim([-1, 1])
        plt.title('data in real time!')
        plt.gcf().autofmt_xdate()
        plt.show()
        plt.pause(0.5)
        time.sleep(0.5)
