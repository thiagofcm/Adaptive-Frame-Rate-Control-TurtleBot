#!/usr/bin/env python3
"""Thin wrapper around the unmodified DRLEnvironment node.

drl_environment.py's own main() rejects any nonzero argv:

    def main(args=sys.argv[1:]):
        rclpy.init(args=args)
        if len(args) == 0:
            drl_environment = DRLEnvironment()
        else:
            rclpy.shutdown()
            quit("ERROR: wrong number of arguments!")

so `ros2 run turtlebot3_drl environment --ros-args -r scan:=scan_gated`
fails outright -- it never expects --ros-args to reach it. This wrapper
calls rclpy.init(args=sys.argv) itself; rclpy/rcl parses and strips
--ros-args before DRLEnvironment is ever constructed.

This file imports DRLEnvironment unmodified and changes NOTHING about
state construction, reward, collision logic, goal logic, services, or
action scaling -- all of that still runs exactly as drl_environment.py
defines it. The only effect of running this instead of
`ros2 run turtlebot3_drl environment` is that the node's 'scan'
subscription can be remapped to a different topic name at launch time,
e.g.:

    python3 environment_gated.py --ros-args -r scan:=scan_gated
"""
import sys

import rclpy
from turtlebot3_drl.drl_environment.drl_environment import DRLEnvironment


def main():
    rclpy.init(args=sys.argv)
    node = DRLEnvironment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
