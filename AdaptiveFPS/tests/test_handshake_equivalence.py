#!/usr/bin/env python3
"""Deterministic call/order tests for the episode handshake of AdaptiveFPSEnv (no Gazebo needed).

ROS is replaced by a scripted fake world:
  - rclpy.spin_once advances a fake clock and delivers /clock, /odom, /scan (and /goal_pose when scheduled)
    through the node's REAL callbacks (gate included) and the env's own /scan callback;
  - service clients (step_comm, pause, unpause, reset_simulation), publishers and the logger are
    recorders; futures complete after a fixed number of spins;
  - time.monotonic follows the fake clock, so timeouts and heartbeats run instantly.
Every spin, callback, service call, publish and log line is appended to one ordered trace.

Modes:
  --reset-trace OUT.json   run full AdaptiveFPSEnv.reset() scenarios, write the traces (compare runs of
                           this mode before/after a refactor)
  --step-trace OUT.json    (a) reset + 30 AdaptiveFPSEnv.step() calls with a scripted rate-action sequence and
                           (b) reset + 300 steps that apply the next rate of 10->5->1->0.5->0.2 Hz at every consumed
                           frame (covers all five gate intervals); records every DrlStep request bitwise (float.hex
                           of action/previous_action, i.e. exactly what DRLEnvironment turns into cmd_vel), every
                           /scan_gated forward, plus the 43-D obs, reward and sensing state per step
  --boundary-trace OUT.json  episode boundaries after SUCCESS / COLLISION / TIMEOUT / TUMBLE: episode 1, run to
                           termination, reset(); checks exactly one normal /reset_simulation per reset(), issued
                           before the unpause, and reports recovery resets separately
  --gate-api               focused checks of GazeboSensingBridge.scan_interval_for / set_sensing_rate / reset_gate /
                           request_fresh_scan against the former direct attribute writes

Needs a sourced ROS 2 workspace, DRLNAV_BASE_PATH and /tmp/drlnav_current_stage.txt (the env loads TD3).
"""
import argparse
import json
import os
import sys
import time
import types

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.environ["DRLNAV_BASE_PATH"])
from AdaptiveFPS.env.adaptive_fps_env import AdaptiveFPSEnv  # noqa: E402


# ---------------------------------------------------------------------------------------------- fake world
class FakeFuture:
    def __init__(self, world, name, delay, result):
        self.world, self.name, self.remaining, self._result = world, name, delay, result

    def done(self):
        return self.remaining <= 0

    def result(self):
        return self._result

    def exception(self):
        return None


class FakeClient:
    def __init__(self, world, name, unavailable_tries=0, delay=2):
        self.world, self.name, self.unavailable_tries, self.delay = world, name, unavailable_tries, delay

    def wait_for_service(self, timeout_sec=None):
        ok = self.unavailable_tries <= 0
        self.unavailable_tries -= 1
        self.world.log("wait_for_service", self.name, timeout_sec, ok)
        return ok

    def call_async(self, req):
        if self.name == "step_comm":
            self.world.step_comm_calls += 1
            summary = [list(req.action), list(req.previous_action)]
            state = [round(0.01 * i, 4) for i in range(44)]
            if self.world.vary_state:   # --step-trace only: a different 44-D state on every call
                self.world.step_calls += 1
                c = self.world.step_calls
                state = [round(((i * 7 + c * 13) % 100) / 100.0, 4) for i in range(40)] + \
                        [round((c % 50) / 50.0, 4), round(((c * 3) % 21 - 10) / 10.0, 4), 0.0, 0.0]
                summary = [[float(x).hex() for x in req.action], [float(x).hex() for x in req.previous_action]]
            done, outcome = False, 0
            if self.world.terminal_at is not None and self.world.step_comm_calls == self.world.terminal_at:
                done, outcome = True, self.world.terminal_outcome
                self.world.log("terminal", outcome)
                self.world.goal_sent, self.world.goal_at = False, self.world.t - 1000.0 + 0.05  # next goal, as gazebo_goals
            result = types.SimpleNamespace(state=state, reward=0.0, done=done, success=outcome, distance_traveled=0.0)
        else:
            summary, result = None, types.SimpleNamespace()
        self.world.log("call_async", self.name, summary)
        fut = FakeFuture(self.world, self.name, self.delay, result)
        self.world.pending.append(fut)
        if self.name == "reset_simulation":
            self.world.on_reset()
        return fut


class FakeLogger:
    def __init__(self, world):
        self.world = world

    def info(self, msg):
        self.world.log("log.info", msg)

    def warning(self, msg):
        self.world.log("log.warning", msg)

    def error(self, msg):
        self.world.log("log.error", msg)


class FakeWorld:
    """Scripted ROS world. goal_at: fake time at which one /goal_pose is delivered (None = never).
    odom_xy/twist: pose reported on /odom; after a reset the pose becomes reset_xy (if given)."""

    def __init__(self, env, goal_at=0.3, odom_xy=(2.5, 2.5), twist=(0.0, 0.0), reset_xy=None,
                 terminal_at=None, terminal_outcome=0):
        self.env, self.node = env, env.node
        self.t, self.sim, self.spins = 1000.0, 0.0, 0
        self.goal_at, self.goal_sent = goal_at, False
        self.odom_xy, self.twist, self.reset_xy = odom_xy, twist, reset_xy
        self.pending, self.trace = [], []
        self.vary_state, self.step_calls = False, 0
        self.step_comm_calls, self.terminal_at, self.terminal_outcome = 0, terminal_at, terminal_outcome

    def log(self, *event):
        self.trace.append([round(self.t - 1000.0, 6)] + [self._clean(e) for e in event])

    @staticmethod
    def _clean(e):
        if isinstance(e, float):
            return round(e, 9)
        return e

    def monotonic(self):
        return self.t

    def on_reset(self):
        self.sim = 0.0
        if isinstance(self.reset_xy, list):      # one pose per reset, in order (last one repeats)
            self.odom_xy = self.reset_xy.pop(0) if len(self.reset_xy) > 1 else self.reset_xy[0]
        elif self.reset_xy is not None:
            self.odom_xy = self.reset_xy

    def spin_once(self, node, timeout_sec=None):
        self.spins += 1
        self.t += timeout_sec if timeout_sec is not None else 0.001
        self.sim += 0.02
        self.log("spin_once", timeout_sec)
        for f in list(self.pending):
            f.remaining -= 1
            if f.done():
                self.pending.remove(f)
        n, env = self.node, self.env
        clock = Clock()
        clock.clock = TimeMsg(sec=int(self.sim), nanosec=int(round((self.sim % 1) * 1e9)))
        n._on_clock(clock)
        odom = Odometry()
        odom.header.stamp = clock.clock
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.odom_xy
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.twist
        n._on_odom(odom)
        scan = LaserScan()
        scan.header.stamp = clock.clock
        scan.ranges = [float(self.spins)] * 40
        n._on_real_scan(scan)
        env._on_native_scan(scan)
        if self.goal_at is not None and not self.goal_sent and self.t - 1000.0 >= self.goal_at:
            goal = Pose()
            goal.position.x, goal.position.y = -1.8, -2.5
            n._on_goal_pose(goal)
            self.goal_sent = True
        self.log("state", n.clock_msg_count, n.odom_msg_count, n.goal_msg_count, n.gated_scan_count,
                 n.scans_since_forward, n.force_next_fresh, n.k)


def install(world, unavailable=None):
    """Point rclpy.spin_once/time.monotonic at the fake world and swap the node's ROS I/O for recorders."""
    unavailable = unavailable or {}
    node = world.node
    node.step_comm_client = FakeClient(world, "step_comm", unavailable.get("step_comm", 0))
    node.gazebo_pause = FakeClient(world, "/pause_physics", unavailable.get("/pause_physics", 0))
    node.gazebo_unpause = FakeClient(world, "/unpause_physics", unavailable.get("/unpause_physics", 0))
    node.reset_simulation_client = FakeClient(world, "reset_simulation", unavailable.get("reset_simulation", 0))
    node.cmd_vel_pub = types.SimpleNamespace(
        publish=lambda m: world.log("publish", "cmd_vel", m.linear.x, m.linear.y, m.linear.z,
                                    m.angular.x, m.angular.y, m.angular.z))
    node.ready_pub = types.SimpleNamespace(publish=lambda m: world.log("publish", "episode_ready"))
    node.gated_pub = types.SimpleNamespace(publish=lambda m: world.log("publish", "scan_gated", m.ranges[0]))
    logger = FakeLogger(world)
    node.get_logger = lambda: logger
    rclpy.spin_once = world.spin_once
    time.monotonic = world.monotonic


def prime(env, **node_state):
    """Identical starting state for every scenario."""
    n = env.node
    defaults = dict(k=5, scans_since_forward=0, force_next_fresh=False, gated_scan_count=0, goal_msg_count=0,
                    latest_goal_xy=None, goal_baseline=0, latest_odom=None, odom_msg_count=0,
                    latest_sim_time=None, clock_msg_count=0, episode_index=0)
    defaults.update(node_state)
    for k, v in defaults.items():
        setattr(n, k, v)
    env._recording_active = False


# ------------------------------------------------------------------------------------------ scenarios
RESET_SCENARIOS = {
    # first episode: robot already at the reset pose
    "first_episode": dict(world=dict(goal_at=0.3)),
    # previous episode ended elsewhere (e.g. at the goal): the episode reset moves the robot back
    "robot_elsewhere_before_reset": dict(world=dict(goal_at=0.3, odom_xy=(-1.7, -2.4), reset_xy=(2.5, 2.5))),
    # the episode reset does not bring the robot back: attempt 1 times out, the recovery reset does, attempt 2 passes
    "recovery_after_failed_attempt": dict(world=dict(goal_at=0.3, odom_xy=(0.0, 0.0),
                                                     reset_xy=[(0.0, 0.0), (2.5, 2.5)])),
    # goal arrives late (heartbeats), services initially unavailable
    "late_goal_services_waiting": dict(world=dict(goal_at=7.5),
                                       unavailable={"/unpause_physics": 2, "step_comm": 1}),
}


def run_reset(env, spec):
    world = FakeWorld(env, **spec["world"])
    prime(env)
    install(world, spec.get("unavailable"))
    try:
        obs, info = env.reset()
        result = {"obs": [round(float(x), 9) for x in obs], "info": {k: v for k, v in info.items()},
                  "exception": None}
    except Exception as exc:  # the trace up to the failure is still compared
        result = {"exception": f"{type(exc).__name__}: {exc}"}
    result["trace"] = world.trace
    result["env_after"] = [env.episode_scan_count, env.prev_gated_scan_count,
                           env.node.goal_baseline, env.node.episode_index]
    return result




STEP_ACTIONS = [4, 4, 0, 0, 3, 1, 2, 4, 4, 3, 3, 2, 1, 0, 4, 2, 2, 3, 1, 4, 0, 1, 2, 3, 4, 4, 1, 0, 3, 2]
STEP_INFO_KEYS = ["frame_consumed", "current_fps", "obs_interval", "episode_scan_count", "scans_since_last_obs",
                  "ppo_step_sim_dt", "goal_distance_m", "goal_distance_norm", "fps_ratio",
                  "obs_age_ratio", "episode_scan_count_ratio", "nav_reward", "frame_penalty",
                  "terminal_reward", "outcome", "x", "y", "yaw", "sim_time"]


RATE_CYCLE = [4, 3, 2, 1, 0]      # action indices: 10, 5, 1, 0.5, 0.2 Hz


def run_steps(env, adaptive_cycle=False, n_steps=None):
    world = FakeWorld(env, goal_at=0.3)
    world.vary_state = True
    prime(env)
    install(world)
    obs, _ = env.reset()
    steps = [{"reset_obs": [float(x).hex() for x in obs]}]
    consumed = 0
    actions = STEP_ACTIONS if not adaptive_cycle else [None] * n_steps
    for a in actions:
        if adaptive_cycle:   # the action only takes effect on a consumed frame: offer the next rate of the cycle
            a = RATE_CYCLE[consumed % len(RATE_CYCLE)]
        obs, reward, terminated, truncated, info = env.step(a)
        consumed += int(bool(info["frame_consumed"]))
        steps.append({"action": a, "obs": [float(x).hex() for x in obs], "reward": float(reward).hex(),
                      "terminated": terminated, "truncated": truncated,
                      "navigation_action": [float(x).hex() for x in info["navigation_action"]],
                      "info": {k: (float(info[k]).hex() if isinstance(info[k], float) else info[k])
                               for k in STEP_INFO_KEYS}})
    return {"steps": steps, "trace": world.trace}


OUTCOMES = {"SUCCESS": 1, "COLLISION_WALL": 2, "COLLISION_OBSTACLE": 3, "TIMEOUT": 4, "TUMBLE": 5}


def reset_segment_stats(trace):
    """Normal vs recovery /reset_simulation calls inside one reset() trace segment, and their order vs unpause."""
    calls = [i for i, e in enumerate(trace) if e[1] == "call_async" and e[2] == "reset_simulation"]
    unpauses = [i for i, e in enumerate(trace) if e[1] == "call_async" and e[2] == "/unpause_physics"]
    recoveries = [i for i, e in enumerate(trace) if e[1] == "log.warning" and "initiating reset recovery" in e[2]]
    normal = [c for c in calls if not any(r < c for r in recoveries)]
    return {"normal_resets": len(normal), "recovery_resets": len(calls) - len(normal),
            "normal_reset_before_first_unpause": bool(normal and unpauses and normal[0] < unpauses[0]),
            "forced_scan_after_reset": bool(normal) and any(
                e[1] == "publish" and e[2] == "scan_gated" for e in trace[normal[0]:])}


def run_boundaries(env):
    out = {}
    for name, code in OUTCOMES.items():
        world = FakeWorld(env, goal_at=0.3, terminal_at=12, terminal_outcome=code)
        prime(env)
        install(world)
        env.reset()
        first_end = len(world.trace)
        terminated, steps = False, 0
        while not terminated and steps < 50:
            _, _, terminated, _, info = env.step(4)
            steps += 1
        second_start = len(world.trace)
        obs, info = env.reset()
        out[name] = {"episode1_reset": reset_segment_stats(world.trace[:first_end]),
                     "terminated_after_steps": steps, "outcome": info.get("outcome", code),
                     "episode2_reset": reset_segment_stats(world.trace[second_start:]),
                     "episode2_trace": world.trace[second_start:]}
    return out


def gate_api_checks(env):
    """GazeboSensingBridge gate API vs. the former direct attribute writes (same expression, same state)."""
    node, ok = env.node, True
    NATIVE = 50.0     # native LiDAR rate the former env-side expression used (NATIVE_SCAN_HZ)
    for fps in [0.2, 0.5, 1.0, 5.0, 10.0, 0.1, 2.0, 3.3, 25.0, 50.0, 100.0]:
        old_k = max(1, round(NATIVE / fps))
        node.k = -1
        k1, k2 = node.scan_interval_for(fps), node.set_sensing_rate(fps)
        same = (k1 == k2 == node.k == old_k) and type(k2) is type(old_k)
        ok &= same
        print(f"  rate {fps:6.2f} Hz -> k={k2:4d} (old expression {old_k:4d}) {'OK' if same else 'MISMATCH'}")
    for force in (True, False):
        node.scans_since_forward, node.gated_scan_count, node.force_next_fresh, node.k = 7, 9, force, 250
        k = node.reset_gate(10.0)
        same = (k, node.k, node.scans_since_forward, node.gated_scan_count, node.force_next_fresh) == (5, 5, 0, 0, force)
        ok &= same
        print(f"  reset_gate(10 Hz) with force_next_fresh={force}: -> k={node.k} counters=({node.scans_since_forward},"
              f"{node.gated_scan_count}) force={node.force_next_fresh} {'OK' if same else 'MISMATCH'}")
    node.force_next_fresh = False
    node.request_fresh_scan()
    ok &= node.force_next_fresh is True
    print(f"  request_fresh_scan() -> force_next_fresh={node.force_next_fresh}")

    # replay the same schedule through the API and through direct writes; compare forwards and counters
    schedule = [(10.0, 30, False), (0.2, 300, False), (0.2, 20, True), (1.0, 50, False), (50.0, 5, False),
                (0.5, 120, False), (1.0, 60, True), (10.0, 12, False), (5.0, 40, False)]

    def replay(use_api):
        published, trace = [], []
        node.gated_pub = types.SimpleNamespace(publish=lambda m: published.append(int(m.ranges[0])))
        if use_api:
            node.reset_gate(10.0)
        else:
            node.k, node.scans_since_forward, node.gated_scan_count = max(1, round(NATIVE / 10.0)), 0, 0
        node.force_next_fresh = True
        i = 0
        for fps, n, force in schedule:
            if use_api:
                node.set_sensing_rate(fps)
                if force:
                    node.request_fresh_scan()
            else:
                node.k = max(1, round(NATIVE / fps))
                if force:
                    node.force_next_fresh = True
            for _ in range(n):
                m = LaserScan()
                m.ranges = [float(i)] * 40
                node._on_real_scan(m)
                trace.append((node.k, node.scans_since_forward, node.gated_scan_count, node.force_next_fresh))
                i += 1
        return published, trace
    api, direct = replay(True), replay(False)
    same = api == direct
    ok &= same
    print(f"  schedule replay ({sum(n for _, n, _ in schedule)} scans): {len(api[0])} forwards, "
          f"API == direct writes: {same}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset-trace")
    ap.add_argument("--step-trace")
    ap.add_argument("--gate-api", action="store_true")
    ap.add_argument("--boundary-trace")
    args = ap.parse_args()

    real_spin_once, real_monotonic = rclpy.spin_once, time.monotonic
    env = AdaptiveFPSEnv()
    try:
        if args.reset_trace:
            traces = {name: run_reset(env, spec) for name, spec in RESET_SCENARIOS.items()}
            for name, r in traces.items():
                print(f"reset scenario {name:32s} events={len(r['trace']):5d} exception={r['exception']}")
            with open(args.reset_trace, "w") as f:
                json.dump(traces, f, indent=1, sort_keys=True, default=str)
        if args.step_trace:
            result = {"scripted": run_steps(env), "rate_cycle": run_steps(env, adaptive_cycle=True, n_steps=300)}
            for name, r in result.items():
                n_req = sum(1 for e in r["trace"] if e[1] == "call_async" and e[2] == "step_comm")
                n_fwd = sum(1 for e in r["trace"] if e[1] == "publish" and e[2] == "scan_gated")
                rates = sorted({s["info"]["obs_interval"] for s in r["steps"][1:]})
                print(f"step trace {name}: {len(r['steps']) - 1} env.step() calls, {n_req} DrlStep requests, "
                      f"{n_fwd} /scan_gated forwards, intervals used {rates}, {len(r['trace'])} events")
            with open(args.step_trace, "w") as f:
                json.dump(result, f, indent=1, sort_keys=True, default=str)
        if args.boundary_trace:
            res = run_boundaries(env)
            ok = True
            for name, r in res.items():
                a, b = r["episode1_reset"], r["episode2_reset"]
                good = all(x["normal_resets"] == 1 and x["recovery_resets"] == 0 and
                           x["normal_reset_before_first_unpause"] and x["forced_scan_after_reset"] for x in (a, b))
                ok &= good and r["terminated_after_steps"] < 50
                print(f"  after {name:18s}: episode 1 reset {a} | terminated after {r['terminated_after_steps']} steps"
                      f" | next reset {b} -> {'OK' if good else 'CHECK'}")
            seqs = {name: [(e[1], e[2]) for e in r["episode2_trace"] if e[1] in ("call_async", "publish")]
                    for name, r in res.items()}
            same = all(v == seqs["SUCCESS"] for v in seqs.values())
            ok &= same
            print(f"  reset() service/publish sequence identical for all outcomes: {same}")
            with open(args.boundary_trace, "w") as f:
                json.dump(res, f, indent=1, sort_keys=True, default=str)
            print("BOUNDARY:", "PASS" if ok else "FAIL")
            if not ok:
                raise SystemExit(1)
        if args.gate_api:
            ok = gate_api_checks(env)
            print("GATE API:", "PASS" if ok else "FAIL")
            if not ok:
                raise SystemExit(1)
    finally:
        rclpy.spin_once, time.monotonic = real_spin_once, real_monotonic
        env.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
