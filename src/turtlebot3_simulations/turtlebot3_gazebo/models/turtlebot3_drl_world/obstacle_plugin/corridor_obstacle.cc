// Parameterised moving-obstacle plugin for Stage 13 (corridor_dynamic_chase).
//
// Reproduces the moving-obstacle specs of adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py.
// Supported <type> values:
//   ping_pong:       <start>x y</start> <end>x y</end> <speed>m/s</speed>
//                    start -> end -> start ..., phase 0 (at start, heading to end) at sim time 0.
//                    The motion is a pure function of Gazebo sim time, which /reset_simulation sets
//                    back to 0, so every reset restarts it from the same initial state.
//   triggered_chase: <start>x y</start> <speed>m/s</speed> <chase_duration>s</chase_duration>
//                    <trigger_x>x</trigger_x> [<trigger_direction>decreasing</trigger_direction>]
//                    [<robot_model>turtlebot3_burger</robot_model>]
//                    Stays at start until the robot's x crosses trigger_x (prev_x > trigger_x >= x,
//                    once per episode), then for chase_duration seconds moves toward the robot's
//                    current Gazebo world (x, y) at `speed` without overshooting, then stops where it
//                    is. Reset() (called by /reset_simulation) returns it to start, untriggered.
//
// Usage (wrap the obstacle body, as the Stage 9 worlds do):
//   <model name="turtlebot3_drl_obstacle1">
//     <pose>1.4 2.0 0 0 0 0</pose>
//     <plugin name="corridor_ob1" filename="libcorridor_obstacle.so">
//       <type>ping_pong</type> <start>1.4 2.0</start> <end>2.4 2.0</end> <speed>0.06</speed>
//     </plugin>
//     <include><uri>model://turtlebot3_drl_world/obstacle1</uri></include>
//   </model>

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector2.hh>
#include <ignition/math/Vector3.hh>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class CorridorObstacle : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      this->model = _model;
      this->world = _model->GetWorld();

      const std::string type = _sdf->Get<std::string>("type", "ping_pong").first;
      if (type == "ping_pong")
      {
        this->type = Type::PING_PONG;
        if (!this->LoadPingPong(_sdf))
          return;
      }
      else if (type == "triggered_chase")
      {
        this->type = Type::TRIGGERED_CHASE;
        if (!this->LoadChase(_sdf))
          return;
      }
      else
      {
        gzerr << "[corridor_obstacle] " << this->model->GetName() << ": unsupported <type> '" << type
              << "', obstacle stays static\n";
        return;
      }

      // Keep the height given in the world file; only x/y are driven.
      this->z = this->model->WorldPose().Pos().Z();

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CorridorObstacle::OnUpdate, this));
      if (this->type == Type::PING_PONG)
        this->Apply(this->world->SimTime().Double());
      else
        this->ResetChase();
    }

    // Called by World::Reset() (/reset_simulation), which also sets sim time back to 0.
    void Reset() override
    {
      if (!this->updateConnection)
        return;
      if (this->type == Type::PING_PONG)
        this->Apply(0.0);
      else
      {
        this->ResetChase();
        this->Log("reset: back at start, untriggered");
      }
    }

  private:
    enum class Type { PING_PONG, TRIGGERED_CHASE };

    bool HasAll(const sdf::ElementPtr &_sdf, std::initializer_list<const char *> keys)
    {
      for (const char *key : keys)
      {
        if (!_sdf->HasElement(key))
        {
          gzerr << "[corridor_obstacle] " << this->model->GetName() << ": missing <" << key
                << ">, obstacle stays static\n";
          return false;
        }
      }
      return true;
    }

    void Log(const std::string &msg)
    {
      std::cout << "[corridor_obstacle] " << this->model->GetName() << " t=" << this->world->SimTime().Double()
                << "s: " << msg << std::endl;
    }

    void OnUpdate()
    {
      if (this->type == Type::PING_PONG)
        this->Apply(this->world->SimTime().Double());
      else
        this->UpdateChase(this->world->SimTime().Double());
    }

    // ------------------------------------------------------------------ ping_pong

    bool LoadPingPong(const sdf::ElementPtr &_sdf)
    {
      if (!this->HasAll(_sdf, {"start", "end", "speed"}))
        return false;
      this->start = _sdf->Get<ignition::math::Vector2d>("start");
      this->end = _sdf->Get<ignition::math::Vector2d>("end");
      this->speed = _sdf->Get<double>("speed");
      this->length = this->start.Distance(this->end);
      if (this->speed <= 0.0 || this->length <= 0.0)
      {
        gzerr << "[corridor_obstacle] " << this->model->GetName()
              << ": ping_pong needs speed > 0 and start != end, obstacle stays static\n";
        return false;
      }
      this->leg = this->length / this->speed;   // seconds for one start -> end pass

      gzmsg << "[corridor_obstacle] " << this->model->GetName() << ": ping_pong " << this->start << " -> "
            << this->end << " at " << this->speed << " m/s (leg " << this->leg << " s)\n";
      return true;
    }

    // Same triangle wave as TurtleBot_VarScanRate._obstacle_positions() for "ping_pong":
    //   leg = |end - start| / speed;  ph = t mod 2*leg
    //   frac = ph / leg (outbound) or (2*leg - ph) / leg (return);  pos = start + (end - start) * frac
    void Apply(double t)
    {
      const double ph = std::fmod(t, 2.0 * this->leg);
      const bool outbound = ph <= this->leg;
      const double frac = (outbound ? ph : 2.0 * this->leg - ph) / this->leg;
      const ignition::math::Vector2d pos = this->start + (this->end - this->start) * frac;
      const ignition::math::Vector2d vel =
          (this->end - this->start) / this->length * this->speed * (outbound ? 1.0 : -1.0);

      this->model->SetWorldPose(ignition::math::Pose3d(pos.X(), pos.Y(), this->z, 0.0, 0.0, 0.0));
      // Kinematic body: overwrite link velocities so gravity/contacts never accumulate, and so
      // the p3d odometry twist reports the commanded ping-pong velocity.
      this->SetVelocity(this->model, ignition::math::Vector3d(vel.X(), vel.Y(), 0.0));
    }

    // ------------------------------------------------------------------ triggered_chase

    bool LoadChase(const sdf::ElementPtr &_sdf)
    {
      if (!this->HasAll(_sdf, {"start", "speed", "chase_duration", "trigger_x"}))
        return false;
      this->start = _sdf->Get<ignition::math::Vector2d>("start");
      this->speed = _sdf->Get<double>("speed");
      this->chaseDuration = _sdf->Get<double>("chase_duration");
      this->triggerX = _sdf->Get<double>("trigger_x");
      const std::string direction = _sdf->Get<std::string>("trigger_direction", "decreasing").first;
      this->robotName = _sdf->Get<std::string>("robot_model", "turtlebot3_burger").first;
      if (direction != "decreasing")
      {
        gzerr << "[corridor_obstacle] " << this->model->GetName()
              << ": triggered_chase only supports trigger_direction 'decreasing', obstacle stays static\n";
        return false;
      }
      if (this->speed <= 0.0 || this->chaseDuration <= 0.0)
      {
        gzerr << "[corridor_obstacle] " << this->model->GetName()
              << ": triggered_chase needs speed > 0 and chase_duration > 0, obstacle stays static\n";
        return false;
      }

      std::cout << "[corridor_obstacle] " << this->model->GetName() << ": triggered_chase start " << this->start
                << ", speed " << this->speed << " m/s for " << this->chaseDuration << " s after robot '"
                << this->robotName << "' x crosses " << this->triggerX << " (decreasing)" << std::endl;
      return true;
    }

    void ResetChase()
    {
      this->triggered = false;
      this->chaseEnded = false;
      this->triggerTime = std::numeric_limits<double>::quiet_NaN();
      this->chaseTimeUsed = 0.0;
      this->prevRobotX = std::numeric_limits<double>::quiet_NaN();
      this->lastTime = this->world->SimTime().Double();
      this->chasePos = this->start;
      this->model->SetWorldPose(
          ignition::math::Pose3d(this->start.X(), this->start.Y(), this->z, 0.0, 0.0, 0.0));
      this->SetVelocity(this->model, ignition::math::Vector3d::Zero);
    }

    // Mirrors TurtleBot_VarScanRate._physics_step(): advance an active chase first
    // (_advance_obstacle_chasers), then check the trigger (_check_obstacle_triggers), so the
    // obstacle does not move in the update that triggers it. Here one "step" is one physics tick.
    void UpdateChase(double t)
    {
      const double dt = std::max(0.0, t - this->lastTime);
      this->lastTime = t;

      if (!this->robot)
      {
        this->robot = this->world->ModelByName(this->robotName);
        if (!this->robot)
        {
          if (!this->robotMissingLogged)
            gzerr << "[corridor_obstacle] " << this->model->GetName() << ": robot model '" << this->robotName
                  << "' not found, holding position\n";
          this->robotMissingLogged = true;
          this->HoldChasePose(ignition::math::Vector2d::Zero);
          return;
        }
      }
      const ignition::math::Vector3d robotPos = this->robot->WorldPose().Pos();
      const ignition::math::Vector2d robotXY(robotPos.X(), robotPos.Y());

      // 1) chase: move toward the robot's current position by min(speed * dt, distance), no overshoot;
      //    the total chase time is capped at chase_duration (Python: round(chase_duration / dt) steps).
      ignition::math::Vector2d vel = ignition::math::Vector2d::Zero;
      if (this->triggered && !this->chaseEnded)
      {
        const double stepTime = std::min(dt, this->chaseDuration - this->chaseTimeUsed);
        const ignition::math::Vector2d d = robotXY - this->chasePos;
        const double dist = d.Length();
        if (stepTime > 0.0 && dist > 0.0)
        {
          const ignition::math::Vector2d move = d * (std::min(this->speed * stepTime, dist) / dist);
          this->chasePos += move;
          vel = move / dt;
        }
        this->chaseTimeUsed += std::max(0.0, stepTime);
        if (this->chaseTimeUsed >= this->chaseDuration - 1e-9)
        {
          this->chaseEnded = true;
          std::ostringstream s;
          s << "chase ended after " << this->chaseTimeUsed << " s at (" << this->chasePos.X() << ", "
            << this->chasePos.Y() << "), robot at (" << robotXY.X() << ", " << robotXY.Y()
            << "); stopping here until reset";
          this->Log(s.str());
        }
      }

      // 2) trigger: robot x crossing trigger_x in the decreasing direction, once per episode.
      if (!this->triggered && std::isfinite(this->prevRobotX) && this->prevRobotX > this->triggerX &&
          robotXY.X() <= this->triggerX)
      {
        this->triggered = true;
        this->triggerTime = t;
        std::ostringstream s;
        s << "TRIGGERED: robot x " << this->prevRobotX << " -> " << robotXY.X() << " crossed trigger_x "
          << this->triggerX << ", robot at (" << robotXY.X() << ", " << robotXY.Y() << "), obstacle at ("
          << this->chasePos.X() << ", " << this->chasePos.Y() << "); chasing for " << this->chaseDuration << " s";
        this->Log(s.str());
      }
      this->prevRobotX = robotXY.X();

      this->HoldChasePose(vel);
    }

    void HoldChasePose(const ignition::math::Vector2d &vel)
    {
      this->model->SetWorldPose(
          ignition::math::Pose3d(this->chasePos.X(), this->chasePos.Y(), this->z, 0.0, 0.0, 0.0));
      this->SetVelocity(this->model, ignition::math::Vector3d(vel.X(), vel.Y(), 0.0));
    }

    // ------------------------------------------------------------------ shared

    // Model::SetLinearVel only touches the model's own links; the obstacle body is a nested model.
    void SetVelocity(const physics::ModelPtr &m, const ignition::math::Vector3d &vel)
    {
      for (const auto &link : m->GetLinks())
      {
        link->SetLinearVel(vel);
        link->SetAngularVel(ignition::math::Vector3d::Zero);
      }
      for (const auto &nested : m->NestedModels())
        this->SetVelocity(nested, vel);
    }

    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;

    Type type = Type::PING_PONG;
    ignition::math::Vector2d start;
    double speed = 0.0;
    double z = 0.0;

    // ping_pong
    ignition::math::Vector2d end;
    double length = 0.0;
    double leg = 0.0;

    // triggered_chase configuration
    double chaseDuration = 0.0;
    double triggerX = 0.0;
    std::string robotName;
    physics::ModelPtr robot;
    bool robotMissingLogged = false;

    // triggered_chase per-episode state (cleared by ResetChase)
    bool triggered = false;
    bool chaseEnded = false;
    double triggerTime = std::numeric_limits<double>::quiet_NaN();
    double chaseTimeUsed = 0.0;
    double prevRobotX = std::numeric_limits<double>::quiet_NaN();
    double lastTime = 0.0;
    ignition::math::Vector2d chasePos;
  };

  GZ_REGISTER_MODEL_PLUGIN(CorridorObstacle)
} // namespace gazebo
