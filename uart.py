import serial


class uart():
    def __init__(self):
        self.ser = serial.Serial(
            port="/dev/ttyAMA0",
            baudrate=9600,  # baud rate波特率
            bytesize=8,  # number of databits数据位
            parity='N',  # enable parity checking
            stopbits=1,  # number of stopbits
            timeout=0.01,  # set a timeout value, None for waiting forever
            xonxoff=0,  # enable software flow control
            rtscts=0,  # enable RTS/CTS flow control
            interCharTimeout=None  # Inter-character timeout, None to disable
        )

    def send_data(self, send_list):
        result = self.ser.write(send_list.encode("ascii"))
        return result

    def read_data(self, data_length):
        data = self.ser.read(data_length)
        return data
