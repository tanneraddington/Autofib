#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <unordered_map>

/////This node is just a publisher to take in keyboard input (up/down) to control gripper motor (ID = 9) to open or close gripper/////
/////Code can be un-commented to control camera motor (ID = 10) in discrete sections (1, 2, 3 keys)/////

class GripperNode : public rclcpp::Node {
    public:
        GripperNode() : Node("gripper_node") {
            gripper_pub = this->create_publisher<std_msgs::msg::Float32>("/gripper_control",10);
            RCLCPP_INFO(this->get_logger(),"Gripper Node started. Use up/down arrow keys to control.");
            run_keyboard_control();
        }
    
    private:
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr gripper_pub;
        //Callback function that constantly waits for keyboard input
        void run_keyboard_control() {
            struct termios oldt, newt;      //storing old terminal settings and newt used to modify settings
            tcgetattr(STDIN_FILENO, &oldt); //retrieves current terminal settings for standard input and save into oldt structure
            newt = oldt;        //copy original settings (we modify newt and leave oldt unchanged)
            newt.c_lflag &= ~(ICANON | ECHO);   //~ICANON = char by char input (enter not required), ~ECHO = hides what user types
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);    //applies modified terminal settings to standard input

            char ch;
            while(rclcpp::ok()) {       //constantly run
                if (read(STDIN_FILENO, &ch,1) > 0) {    //read 1 byte and store as ch and check if it exists
                    if (ch == 27) { //if special key (arrow keys are)
                        char seq[2];    //1st character in special keys is ESC, so read second character to see which special ky
                        if (read(STDIN_FILENO, &seq[0],1) > 0 && read(STDIN_FILENO, &seq[1],1) > 0) {
                            if (seq[0] =='[') {
                                if (seq[1] == 'A') {    //up arrow
                                    publish_gripper_position(0.008);
                                } else if (seq[1] =='B') {  //down arrow
                                    publish_gripper_position(-0.008);
                                }
                            }
                        }
                    // Move camera & arms to zone 1 of workspace    
                    } 
                    //Controls camera in discrete sections
                    // else if (ch == '1') {     
                    //     publish_gripper_position(1.0);
                    // // Move camera & arms to zone 2 of workspace 
                    // } else if (ch == '2') {
                    //     publish_gripper_position(2.0);
                    // // Move camera & arms to zone 3 of workspace 
                    // } else if (ch =='3') {
                    //     publish_gripper_position(3.0);
                    // } 
                    //Quit
                    else if (ch == 'q') {
                        break;
                    }
                }
            }

            tcsetattr(STDIN_FILENO,TCSANOW, &oldt);     //apply old terminal settings again to restart
        }

        void publish_gripper_position(float value) {
            auto msg = std_msgs::msg::Float32();
            msg.data = value;
            //Publishes .008 to open gripper and -.008 to close gripper (handled in motor node)
            gripper_pub->publish(msg);
            RCLCPP_INFO(this->get_logger(),"Sent Command: %.2f",msg.data);
        }
};

int main(int argc, char **argv) {
    rclcpp::init(argc,argv);
    auto node = std::make_shared<GripperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}