// Randomized ping-pong obstacle plugin for Stage 14 (corridor_dynamic_chase_random).
//
// Reproduces the "random_ping_pong" moving-obstacle spec of
// adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py (motion logic in random_ping_pong.hh):
//   <start>x y</start> <end>x y</end> <speed>m/s</speed>
//   [<reversal_start>0 0.25</reversal_start>] [<reversal_end>0.75 1</reversal_end>]   fractions of the path
//   [<obstacle_index>0</obstacle_index>]   selects the random stream, must differ between obstacles
//   [<seed>0</seed>]                       overridden by the environment variable CORRIDOR_OBSTACLE_SEED
//
// A new realization (start position, direction, reversal targets) is drawn at Load() (episode 0) and at
// every Reset() (/reset_simulation; episode 1, 2, ...) from (seed, episode, obstacle_index), so a run is
// reproducible from its seed and the number of resets. Reset() is the only place that randomizes.
//
// Sim time is not assumed monotonic: /reset_simulation sets it back to 0. The first update after Load() or
// Reset() only records the time, and a negative or oversized time difference never moves the obstacle by
// more than kMaxDt of travel.
//
// If CORRIDOR_OBSTACLE_LOG is set, every reset and reversal is appended to that CSV file:
//   event,model,index,seed,episode,sim_time,s,x,y,dir,target
//
// Usage (wrap the obstacle body, as the Stage 13 world does):
//   <model name="turtlebot3_drl_obstacle1">
//     <pose>1.4 2.0 0 0 0 0</pose>
//     <plugin name="corridor_ob1" filename="libcorridor_random_obstacle.so">
//       <start>1.4 2.0</start> <end>2.4 2.0</end> <speed>0.06</speed> <obstacle_index>0</obstacle_index>
//     </plugin>
//     <include><uri>model://turtlebot3_drl_world/obstacle1</uri></include>
//   </model>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector2.hh>
#include <ignition/math/Vector3.hh>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include "random_ping_pong.hh"

namespace gazebo
{
  namespace
  {
    const double kMaxDt = 0.1;   // s, largest time difference integrated in one update

    // One event log shared by all obstacles of the world (same gzserver process).
    std::mutex logMutex;
    std::ofstream logFile;
    bool logOpened = false;

    void WriteLog(const std::string &line)
    {
      std::lock_guard<std::mutex> lock(logMutex);
      if (!logOpened)
      {
        logOpened = true;
        const char *path = std::getenv("CORRIDOR_OBSTACLE_LOG");
        if (path && *path)
        {
          const bool empty = !std::ifstream(path).good() ||
                             std::ifstream(path).peek() == std::ifstream::traits_type::eof();
          logFile.open(path, std::ios::app);
          if (!logFile)
            gzerr << "[corridor_random_obstacle] cannot open event log '" << path << "'\n";
          else if (empty)
            logFile << "event,model,index,seed,episode,sim_time,s,x,y,dir,target" << std::endl;
        }
      }
      if (logFile)
        logFile << line << std::endl;
    }
  }

  class CorridorRandomObstacle : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      this->model = _model;
      this->world = _model->GetWorld();

      for (const char *key : {"start", "end", "speed"})
      {
        if (!_sdf->HasElement(key))
        {
          gzerr << "[corridor_random_obstacle] " << this->model->GetName() << ": missing <" << key
                << ">, obstacle stays static\n";
          return;
        }
      }
      this->start = _sdf->Get<ignition::math::Vector2d>("start");
      this->end = _sdf->Get<ignition::math::Vector2d>("end");
      const double speed = _sdf->Get<double>("speed");
      const auto revStart =
          _sdf->Get<ignition::math::Vector2d>("reversal_start", ignition::math::Vector2d(0.0, 0.25)).first;
      const auto revEnd =
          _sdf->Get<ignition::math::Vector2d>("reversal_end", ignition::math::Vector2d(0.75, 1.0)).first;
      this->index = _sdf->Get<std::uint64_t>("obstacle_index", 0).first;
      this->seed = _sdf->Get<std::uint64_t>("seed", 0).first;
      const char *envSeed = std::getenv("CORRIDOR_OBSTACLE_SEED");
      if (envSeed && *envSeed)
        this->seed = std::strtoull(envSeed, nullptr, 10);

      if (!this->motion.Configure(this->start.Distance(this->end), speed, revStart.X(), revStart.Y(),
                                  revEnd.X(), revEnd.Y()))
      {
        gzerr << "[corridor_random_obstacle] " << this->model->GetName()
              << ": needs start != end, speed > 0, reversal_start=(0, f1), reversal_end=(f2, 1) with f1 < f2,"
              << " obstacle stays static\n";
        return;
      }

      // Keep the height given in the world file; only x/y are driven.
      this->z = this->model->WorldPose().Pos().Z();

      std::cout << "[corridor_random_obstacle] " << this->model->GetName() << ": random_ping_pong "
                << this->start << " <-> " << this->end << " at " << speed << " m/s, reversal regions ("
                << revStart << ") (" << revEnd << "), seed " << this->seed << ", index " << this->index
                << std::endl;

      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&CorridorRandomObstacle::OnUpdate, this));
      this->StartEpisode();
    }

    // Called by World::Reset() (/reset_simulation), which also sets sim time back to 0.
    void Reset() override
    {
      if (!this->updateConnection)
        return;
      ++this->episode;
      this->StartEpisode();
    }

  private:
    void StartEpisode()
    {
      this->motion.Reset(this->seed, this->episode, this->index);
      this->haveLastTime = false;
      this->ApplyPose();
      this->Event("reset", this->world->SimTime().Double());
      std::cout << "[corridor_random_obstacle] " << this->model->GetName() << " episode " << this->episode
                << ": s0 " << this->motion.S() << " m, dir " << this->motion.Dir() << ", first target "
                << this->motion.Target() << " m" << std::endl;
    }

    void OnUpdate()
    {
      const double t = this->world->SimTime().Double();
      double dt = 0.0;
      if (this->haveLastTime && t > this->lastTime)
        dt = std::min(t - this->lastTime, kMaxDt);
      this->lastTime = t;
      this->haveLastTime = true;

      const double speed = this->motion.Speed();
      this->motion.Advance(dt, [&](double, int, double, double leftover) {
        this->Event("reversal", t - leftover / speed);
      });
      this->ApplyPose();
    }

    ignition::math::Vector2d XY(double s) const
    {
      return this->start + (this->end - this->start) * (s / this->motion.Length());
    }

    void ApplyPose()
    {
      const ignition::math::Vector2d pos = this->XY(this->motion.S());
      const ignition::math::Vector2d vel =
          (this->end - this->start) / this->motion.Length() * this->motion.Speed() * this->motion.Dir();
      this->model->SetWorldPose(ignition::math::Pose3d(pos.X(), pos.Y(), this->z, 0.0, 0.0, 0.0));
      // Kinematic body: overwrite link velocities so gravity/contacts never accumulate, and so
      // the p3d odometry twist reports the commanded velocity.
      this->SetVelocity(this->model, ignition::math::Vector3d(vel.X(), vel.Y(), 0.0));
    }

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

    // Current state after a reset or reversal: s, world (x, y), new direction, new target.
    void Event(const char *event, double simTime)
    {
      const ignition::math::Vector2d pos = this->XY(this->motion.S());
      std::ostringstream line;
      line << std::setprecision(17) << event << ',' << this->model->GetName() << ',' << this->index << ','
           << this->seed << ',' << this->episode << ',' << simTime << ',' << this->motion.S() << ','
           << pos.X() << ',' << pos.Y() << ',' << this->motion.Dir() << ',' << this->motion.Target();
      WriteLog(line.str());
    }

    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;

    ignition::math::Vector2d start;
    ignition::math::Vector2d end;
    double z = 0.0;
    std::uint64_t seed = 0;
    std::uint64_t index = 0;

    corridor::RandomPingPong motion;
    std::uint64_t episode = 0;   // 0 at Load(), +1 at every Reset()
    double lastTime = 0.0;
    bool haveLastTime = false;
  };

  GZ_REGISTER_MODEL_PLUGIN(CorridorRandomObstacle)
} // namespace gazebo
