from serial import Serial
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

#Establishes serial connection to Virtuoso UI

class UINodeL(Node):
    def __init__(self):
        super().__init__('ui_publisher_node_l')

        self.publisher_ = self.create_publisher(Twist,'ui_twist_l',10)

        self.serial_port = '/dev/ttyACM1'   #Make sure this matches your USB port for controller
        self.baud_rate = 115200
        self.ser = Serial(self.serial_port,self.baud_rate,timeout=1)    

        self.timer = self.create_timer(0.0005,self.read_serial_data)        #Publishing rate

        self.get_logger().info('UI Publisher Node Initialized')

    def read_serial_data(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').strip()
            self.get_logger().info(f'Received: {data}')

            try:
                #x,y,z,w = map(float,data.split('\t'))
                x,y,z,w = map(float,data.split(','))
                twist_msg = Twist()
                twist_msg.linear.x = x
                twist_msg.linear.y = y
                twist_msg.linear.z = z
                twist_msg.angular.z = w

                #Publish to ROS2 topic
                self.publisher_.publish(twist_msg)
                self.get_logger().info('Published')

            except ValueError as e:
                self.get_logger().error(f'Error parsing data: {e}')

    def destroy_node(self):
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
        rclpy.init(args=args)

        ui_node = UINodeL()

        rclpy.spin(ui_node)

        ui_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()