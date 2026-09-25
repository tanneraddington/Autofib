import os
import select
import sys
import rclpy

from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.0

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

msg = """
Control Your Robot!
---------------------------
Moving around:
        w         r
   a    s    d    f
        x

w/x : increase/decrease linear velocity in y
a/d : increase/decrease linear velocity in x
r/f : increase/decrease linear velocity in z

space key, s : force stop

CTRL-C to quit
"""

e = """
Communications Failed
"""

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()

    qos = QoSProfile(depth=10)
    node = rclpy.create_node('teleop_keyboard')
    pub = node.create_publisher(Twist, 'teleop/input/twist', qos)

    status = 0
    target_linear_x = 0.0
    target_linear_y = 0.0
    target_linear_z = 0.0
    control_linear_x = 0.0
    control_linear_y = 0.0
    control_linear_z = 0.0

    try:
        print(msg)
        while(1):
            key = get_key(settings)
            if key == 'w':
                target_linear_y = constrain(target_linear_y + LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == 'x':
                target_linear_y = constrain(target_linear_y - LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == 'a':
                target_linear_x = constrain(target_linear_x + LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == 'd':
                target_linear_x = constrain(target_linear_x - LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == 'r':
                target_linear_z = constrain(target_linear_z + LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == 'f':
                target_linear_z = constrain(target_linear_z - LIN_VEL_STEP_SIZE, -MAX_LIN_VEL, MAX_LIN_VEL)
                status = status + 1
            elif key == ' ' or key == 's':
                target_linear_x = 0.0
                control_linear_x = 0.0
                target_linear_y = 0.0
                control_linear_y = 0.0
                target_linear_z = 0.0
                control_linear_z = 0.0
            else:
                if (key == '\x03'):
                    break

            if status == 20:
                print(msg)
                status = 0

            twist = Twist()

            control_linear_x = target_linear_x
            twist.linear.x = control_linear_x
            control_linear_y = target_linear_y
            twist.linear.y = control_linear_y
            control_linear_z = target_linear_z
            twist.linear.z = control_linear_z

            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        pub.publish(twist)

        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':
    main()