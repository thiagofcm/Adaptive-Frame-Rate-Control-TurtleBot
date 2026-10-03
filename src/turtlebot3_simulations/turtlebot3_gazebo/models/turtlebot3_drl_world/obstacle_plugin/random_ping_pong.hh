// random_ping_pong motion of adaptive_sensor_policy/envs/turtlebot_var_scan_rate.py
// (_reset_random_ping_pong, _rpp_sample_target, _advance_random_ping_pong), with no Gazebo dependency so the
// same code runs in the corridor_random_obstacle plugin (Stage 14) and in the offline trace harness
// (tools/stage14/random_ping_pong_trace.cc).
//
// Path coordinate s in [0, L] (metres from `start`), direction +1 toward `end` / -1 toward `start`.
// At every Reset: s0 ~ U(0, L), dir ~ {-1, +1}, first target uniform in the part of the reversal region
// ahead of s. Every reversal happens exactly at the target, draws a new target from the opposite region and
// spends the leftover distance toward it (no pause, no lost distance).
//
// Random numbers: a private mt19937_64 seeded from (seed, episode, obstacle index), fresh at every Reset
// (the counterpart of np.random.default_rng([obstacle_seed, i])). Same distribution as the Python
// environment, not the same numbers. seed_seq, mt19937_64 and Unit() are fully specified, so a
// (seed, episode, index) triple gives the same realization on every platform.

#ifndef CORRIDOR_RANDOM_PING_PONG_HH_
#define CORRIDOR_RANDOM_PING_PONG_HH_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>

namespace corridor
{
  class RandomPingPong
  {
  public:
    // Reversal regions as fractions of L: start region (f0, f1), end region (f2, f3).
    // Same constraint as the Python constructor: speed > 0, f0 == 0 <= f1 < f2 <= f3 == 1.
    bool Configure(double _length, double _speed, double f0, double f1, double f2, double f3)
    {
      if (!(_length > 0.0 && _speed > 0.0 && f0 == 0.0 && f0 <= f1 && f1 < f2 && f2 <= f3 && f3 == 1.0))
        return false;
      this->length = _length;
      this->speed = _speed;
      this->startLo = f0 * _length;
      this->startHi = f1 * _length;
      this->endLo = f2 * _length;
      this->endHi = f3 * _length;
      return true;
    }

    // Testing only: replace the generator by an external source of U[0, 1) draws.
    void SetUnitSource(std::function<double()> source) { this->unitSource = std::move(source); }

    void Reset(std::uint64_t seed, std::uint64_t episode, std::uint64_t index)
    {
      std::seed_seq seq{Lo(seed), Hi(seed), Lo(episode), Hi(episode), Lo(index), Hi(index)};
      this->rng.seed(seq);
      this->s = this->Uniform(0.0, this->length);
      this->dir = this->Unit() < 0.5 ? 1 : -1;
      this->target = this->SampleTarget();
    }

    // Advance by exactly speed * dt of path distance. onReversal(s, newDir, newTarget, leftover) is called
    // at each reversal; leftover is the distance still to be spent in this call after the reversal.
    template <typename F>
    void Advance(double dt, F &&onReversal)
    {
      double remaining = this->speed * dt;
      while (remaining > 0.0)
      {
        const double gap = std::abs(this->target - this->s);
        if (remaining < gap)
        {
          this->s += this->dir * remaining;
          break;
        }
        remaining -= gap;
        this->s = this->target;
        this->dir = -this->dir;
        this->target = this->SampleTarget();
        onReversal(this->s, this->dir, this->target, remaining);
      }
    }

    void Advance(double dt)
    {
      this->Advance(dt, [](double, int, double, double) {});
    }

    double S() const { return this->s; }
    int Dir() const { return this->dir; }
    double Target() const { return this->target; }
    double Length() const { return this->length; }
    double Speed() const { return this->speed; }

  private:
    static std::uint32_t Lo(std::uint64_t v) { return static_cast<std::uint32_t>(v); }
    static std::uint32_t Hi(std::uint64_t v) { return static_cast<std::uint32_t>(v >> 32); }

    // U[0, 1) with 53 random bits.
    double Unit()
    {
      if (this->unitSource)
        return this->unitSource();
      return static_cast<double>(this->rng() >> 11) * (1.0 / 9007199254740992.0);
    }

    double Uniform(double lo, double hi) { return lo + (hi - lo) * this->Unit(); }

    // Uniform target in the part of the reversal region (toward the current direction) that lies ahead of s.
    double SampleTarget()
    {
      if (this->dir > 0)
        return this->Uniform(std::max(this->endLo, this->s), this->endHi);
      return this->Uniform(this->startLo, std::min(this->startHi, this->s));
    }

    double length = 0.0;
    double speed = 0.0;
    double startLo = 0.0, startHi = 0.0, endLo = 0.0, endHi = 0.0;

    double s = 0.0;
    int dir = 1;
    double target = 0.0;

    std::mt19937_64 rng;
    std::function<double()> unitSource;
  };
} // namespace corridor

#endif
