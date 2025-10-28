import signal
import sys
import threading
import os
import csv
import json
import pyqtgraph.exporters

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('loss_graph')

        self.window = window

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'loss',
            self.data_callback,
            10
        )
        self.subscription

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('Loss')
        self.setGeometry(50, 50, 600, 650)

        self.ep = []
        self.loss_list = []
        self.epsilon_list = []
        self.count = 1

        self.plot()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg):
        self.loss_list.append(msg.data[0])
        self.ep.append(msg.data[1])
        self.epsilon_list.append(msg.data[2])
        self.count += 1

    def plot(self):
        self.lossPlt = pyqtgraph.PlotWidget(self, title='Loss')
        self.lossPlt.setGeometry(0, 320, 600, 300)

        self.epsilonPlt = pyqtgraph.PlotWidget(self, title='Epsilon')
        self.epsilonPlt.setGeometry(0, 10, 600, 300)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.lossPlt.showGrid(x=True, y=True)
        self.epsilonPlt.showGrid(x=True, y=True)
        self.lossPlt.plot(self.ep, self.loss_list, pen=(255, 0, 0), clear=True)
        self.epsilonPlt.plot(self.ep, self.epsilon_list, pen=(0, 255, 0), clear=True)


    def save_graphs(self, folder_path):
        self.lossPlt.plot(self.ep, self.loss_list, pen=(255, 0, 0), clear=True)
        self.epsilonPlt.plot(self.ep, self.epsilon_list, pen=(0, 255, 0), clear=True)

        export_loss = pyqtgraph.exporters.ImageExporter(self.lossPlt.plotItem)
        export_loss.parameters()['width'] = 600
        export_loss.export(os.path.join(folder_path,'loss.png'))

        export_epsilon = pyqtgraph.exporters.ImageExporter(self.epsilonPlt.plotItem)
        export_epsilon.parameters()['width'] = 600
        export_epsilon.export(os.path.join(folder_path,'epsilon.png'))

    def save_data_csv(self, folder_path):
        self.lossPlt.plot(self.ep, self.loss_list, pen=(255, 0, 0), clear=True)
        self.epsilonPlt.plot(self.ep, self.epsilon_list, pen=(0, 255, 0), clear=True)

        with open(os.path.join(folder_path,'loss_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Loss','Epsilon'])
            for ep, l, e in zip(self.ep, self.loss_list, self.epsilon_list):
                writer.writerow([ep, l, e])

    def closeEvent(self, event):

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'),'r') as temp:
            data  = json.load(temp)
            folder_name = data['folder']
        #os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'))

        self.save_graphs(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done'),folder_name))
        self.save_data_csv(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done'),folder_name))

        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()




def main():

    #initialize ros client library
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'),'r') as temp:
            data  = json.load(temp)
            folder_name = data['folder']
        #os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),'temp.json'))

        win.save_graphs(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done'),folder_name))
        win.save_data_csv(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'trainings_done'),folder_name))

        if win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


    sys.exit(app.exec())


if __name__ == '__main__':
    main()