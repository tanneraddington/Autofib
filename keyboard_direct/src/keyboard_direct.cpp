#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <unordered_map>
#include <cmath>

// Controls motors with keyboard commands

//------------ Key for Controls ------------//
/// Left Arm ///
// q: rotate outer tube CCW
// a: rotate outer tube CW
// w: translate outer tube out
// s: translate outer tube in
// e: translate inner tube out
// d: translate inner tube in
// r: rotate inner tube CCW
// f: rotate inner tube CW
/// Right Arm ///
// y: rotate outer tube CCW
// h: rotate outer tube CW
// u: translate outer tube out
// j: translate outer tube in
// i: translate inner tube out
// k: translate inner tube in
// o: rotate inner tube CCW
// l: rotate inner tube CW
//------------------------------------------//

//NOTE: Each keyboard input is in mm or radians. They keys will either translate tubes by 0.5mm or rotate them by ~5deg

class KeyboardDirect:public rclcpp::Node{
  public:
    KeyboardDirect():Node("keyboard_direct"){
      publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>("/keyboard_control",10);
      // Timer to control publishing rate to motors (can't hold key down)
      timer = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&KeyboardDirect::publish_keyboard_input,this)
      );

      // Initialize values to false upon start up of node
      initial_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      key_status = {
        {"y",false},{"h",false},{"u",false},{"j",false},
        {"i",false},{"k",false},{"o",false},{"l",false},
        {"q",false},{"a",false},{"w",false},{"s",false},
        {"e",false},{"d",false},{"r",false},{"f",false},
        {"\x1B[A",false},{"\x1B[B",false}
      };

      init_keyboard();
    }

    ~KeyboardDirect() {
      reset_keyboard();
    }

  private:
    std::array<float,8>initial_values;
    // std::unordered_map<char,bool> key_status;
    std::unordered_map<std::string, bool> key_status;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher;
    rclcpp::TimerBase::SharedPtr timer;
    struct termios original_terminal_settings_;

    // Change to false if you want to allow outer tubes to move without also moving inner tubes
    bool relative_motion = true;

    // Check for keypress on keyboard
    // char get_keypress() {
    //   char buf = 0;
    //   if (read(0,&buf,1) < 0) {
    //     perror("Read failed");
    //   }
    //   return buf;
    // }
    std::string get_keypress() {
    char buf[3];
    int n = read(0, &buf[0], 1);  // Read first char

    if (n < 0) {
      perror("Read failed");
      return "";
    }

    // Check if it's an escape character
    if (buf[0] == 27) {
      // Try to read two more characters
      if (read(0, &buf[1], 2) < 0) {
        perror("Read failed");
        return "";
      }
      return std::string(buf, 3);  // Return 3-character escape sequence
    } else {
      return std::string(1, buf[0]);  // Return single character
    }
  }


    void publish_keyboard_input() {
      std::string c = get_keypress();

      for (auto &key : key_status) {
        key.second = false;
      }

      if (!c.empty()) {
        if (key_status.find(c) != key_status.end()) {
          key_status[c] = !key_status[c];
        }
      }
      ///////////////////////////added this one line//////////////////////////////////
      initial_values = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      // If inner tube should translate with outer tube
      if (relative_motion) {
        // Right outer & inner translation
        if (key_status["u"]) {
          initial_values[0] = 0.0005f;
          initial_values[2] = 0.0005f;
        }
        if (key_status["j"]) {
          initial_values[0] = -0.0005f;
          initial_values[2] = -0.0005f;
        }
        //Left outer & inner translation
        if (key_status["w"]) {
          initial_values[4] = 0.0005f;
          initial_values[6] = 0.0005f;
        }
        if (key_status["s"]) {
          initial_values[4] = -0.0005f;
          initial_values[6] = -0.0005f;
        }
      }
      // If inner tube translation independent of outer tube translation
      else {
        // Right outer translation
        if (key_status["u"]) initial_values[0] = 0.0005f;
        if (key_status["j"]) initial_values[0] = -0.0005f;
        // Left outer translation
        if (key_status["w"]) initial_values[4] = 0.0005f;
        if (key_status["s"]) initial_values[4] = -0.0005f;

      }
      //Right arm rest of commands other than outer translation
      if (key_status["i"]) initial_values[2] = 0.0005f;     //trans right arm inner tube
      if (key_status["k"]) initial_values[2] = -0.0005f;     
      if (key_status["y"]) initial_values[1] = 0.08f;       //rot right arm outer tube
      if (key_status["h"]) initial_values[1] = -0.08f; 
      if (key_status["o"]) initial_values[3] = 0.08f;       //rot right arm inner tube
      if (key_status["l"]) initial_values[3] = -0.08f;      
      //Left arm rest of commands other than outer translation
      if (key_status["e"]) initial_values[6] = 0.00105f;     //trans left arm inner tube
      if (key_status["d"]) initial_values[6] = -0.0005f;     
      if (key_status["q"]) initial_values[5] = 0.08f;       //rot left arm outer tube
      if (key_status["a"]) initial_values[5] = -0.08f; 
      if (key_status["r"]) initial_values[7] = 0.08f;       //rot left arm inner tube
      if (key_status["f"]) initial_values[7] = -0.08f;   
      //Automatic helical tool insertion (4 CW rotations). Arrow up to insert and arrow down to resect
      if (key_status["\x1B[A"]) initial_values[7] = 4*M_PI;
      if (key_status["\x1B[B"]) initial_values[7] = -4*M_PI;

      // Publish message to motors
      std_msgs::msg::Float32MultiArray msg;
      msg.data = {initial_values[0],initial_values[1], initial_values[2], initial_values[3],initial_values[4],initial_values[5],initial_values[6],initial_values[7]};
      publisher->publish(msg);

      std::cout <<"\rR Outer Rot: " << initial_values[1]
                <<" | R Outer Trans: " << initial_values[3]
                <<" | R Inner Rot: " << initial_values[0]
                <<" | R Inner Trans: " << initial_values[2] 
                <<"\nL Outer Rot: " << initial_values[5]
                <<" | L Outer Trans: " << initial_values[7]
                <<" | L Inner Rot: " << initial_values[4]
                <<" | L Inner Trans: " << initial_values[6] 
                <<"\n"
                <<"-----------------------------------------------------"
                <<" \n ";

      std::cout.flush();
    }

    void init_keyboard() {
      struct termios new_settings;
      tcgetattr(0, &original_terminal_settings_);
      new_settings = original_terminal_settings_;
      new_settings.c_lflag &= ~ICANON;  //disable canonical mode
      new_settings.c_lflag &= ~ECHO;     //disable echo
      tcsetattr(0,TCSANOW,&new_settings);
    }

    void reset_keyboard() {
      tcsetattr(0,TCSANOW, &original_terminal_settings_);
    }
};

int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  auto node = std::make_shared<KeyboardDirect>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
