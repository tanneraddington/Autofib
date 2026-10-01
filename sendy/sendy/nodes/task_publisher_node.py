#!/usr/bin/env python3
"""
UterineTaskPublisher

Same manual state-machine pattern as the ALISS TaskPublisher: type a task
name (with tab-autocomplete) and it gets published on /current_task for every
other node to react to. This is the single dispatcher for the uterine
pipeline — nothing else decides when to run.

Tasks:
    resect_start   right arm — diffusion approach trajectory (uterine_diffusion_node)
    resect         right arm — diffusion resection trajectory (uterine_diffusion_node)
    retract_start  left arm  — diffusion approach trajectory (uterine_diffusion_node)
    screw          left arm  — screw / 2mm insert / screw sequence (uterine_move_node), not model-driven
    retract        left arm  — diffusion retraction trajectory (uterine_diffusion_node)
    calibrate      both arms — start the next arm of the smoother calibration trajectory
                               (mover_uterus run_calibration, same as /mover_uterus/continue)

Type the same task multiple times to repeat it, e.g.:
    retract_start
    screw
    retract
    retract
    retract
"""

import readline
import rclpy
import rclpy.executors
from rclpy.node import Node
from std_msgs.msg import String
import threading


class UterineTaskPublisher(Node):
    def __init__(self):
        super().__init__("uterine_task_publisher")

        self.tasks = [
            "resect_start",
            "resect",
            "retract_start",
            "screw",
            "retract",
            "calibrate",
        ]

        self.publisher_ = self.create_publisher(String, 'current_task', 10)

    def publish_task(self, task_text):
        msg = String()
        msg.data = task_text if task_text else "idle"
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

    def _completer(self, text, state):
        text = text.strip().lower()
        # see https://docs.python.org/3/library/readline.html#readline.set_completer
        possible_tasks = [t for t in self.tasks if t.startswith(text)]
        return possible_tasks[state]

    def _configure_completion(self):
        # treat whitespace as word-break
        readline.set_completer_delims(" \n")
        # install custom completer hook
        readline.set_completer(self._completer)
        # magic string to enable tab-completion
        readline.parse_and_bind("tab: complete")

    def run(self):
        """Repeatedly ask user which task to run and publish it"""

        self._configure_completion()

        self.get_logger().info("UterineTaskPublisher node started")

        print("Enter tasks, or 'exit'/'quit' to exit")
        print(f"   Available tasks: {', '.join(self.tasks)}")
        print("   Now with autocomplete - try pressing tab to check it out!")
        print()

        while True:
            try:
                user_input = input("Enter task to publish: ").strip().lower()
                if not user_input:
                    continue

                if user_input in ["exit", "quit"]:
                    break

                if user_input not in self.tasks:
                    print(f"Task \"{user_input}\" not found, available tasks are: {', '.join(self.tasks)}")
                    continue

                self.publish_task(user_input)
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+D and Ctrl+C
                print("\n")  # End line started by input(...)
                break
            except Exception as e:
                self.get_logger().error(f"Input error: {e}")


def main(args=None):
    rclpy.init(args=args)
    task_publisher = UterineTaskPublisher()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(task_publisher)

    # Prevent annoying "ExternalShutdownException" on Ctrl+C
    def spin():
        try:
            executor.spin()
        except rclpy.executors.ExternalShutdownException:
            pass

    executor_thread = threading.Thread(target=spin, daemon=True)
    executor_thread.start()

    task_publisher.run()

    rclpy.try_shutdown()
    executor_thread.join()


if __name__ == '__main__':
    main()
