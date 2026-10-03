// Offline harness for random_ping_pong.hh (no Gazebo). Used by tools/stage14/validate_offline.py.
//
//   trace  <seed> <episodes> <duration_s> <dt_s>
//       For obstacle index 0 and 1 and episode 0..episodes-1: reset, advance duration/dt steps of dt.
//       CSV on stdout: event,index,episode,t,s,dir,target   (event = reset | reversal | end)
//   replay <length> <speed> <dt_s> <steps>
//       U[0, 1) draws are read from stdin instead of the generator (s0, direction, then one per target).
//       Prints s after the reset and after every step, one per line.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "random_ping_pong.hh"

int main(int argc, char **argv)
{
  if (argc == 6 && !std::strcmp(argv[1], "trace"))
  {
    const std::uint64_t seed = std::strtoull(argv[2], nullptr, 10);
    const long episodes = std::atol(argv[3]);
    const double duration = std::atof(argv[4]), dt = std::atof(argv[5]);
    const long steps = std::lround(duration / dt);
    std::printf("event,index,episode,t,s,dir,target\n");
    for (std::uint64_t index = 0; index < 2; ++index)
    {
      corridor::RandomPingPong m;
      m.Configure(1.0, 0.06, 0.0, 0.25, 0.75, 1.0);
      for (long ep = 0; ep < episodes; ++ep)
      {
        m.Reset(seed, ep, index);
        std::printf("reset,%lu,%ld,0,%.17g,%d,%.17g\n", index, ep, m.S(), m.Dir(), m.Target());
        for (long k = 1; k <= steps; ++k)
        {
          const double t = k * dt;
          m.Advance(dt, [&](double s, int dir, double target, double leftover) {
            std::printf("reversal,%lu,%ld,%.17g,%.17g,%d,%.17g\n", index, ep, t - leftover / m.Speed(), s, dir,
                        target);
          });
          if (!(m.S() >= 0.0 && m.S() <= m.Length()))
          {
            std::fprintf(stderr, "s out of bounds: %.17g\n", m.S());
            return 2;
          }
        }
        std::printf("end,%lu,%ld,%.17g,%.17g,%d,%.17g\n", index, ep, steps * dt, m.S(), m.Dir(), m.Target());
      }
    }
    return 0;
  }
  if (argc == 6 && !std::strcmp(argv[1], "replay"))
  {
    corridor::RandomPingPong m;
    if (!m.Configure(std::atof(argv[2]), std::atof(argv[3]), 0.0, 0.25, 0.75, 1.0))
      return 2;
    const double dt = std::atof(argv[4]);
    const long steps = std::atol(argv[5]);
    m.SetUnitSource([]() {
      double u;
      if (!(std::cin >> u))
      {
        std::fprintf(stderr, "ran out of draws\n");
        std::exit(3);
      }
      return u;
    });
    m.Reset(0, 0, 0);
    std::printf("%.17g\n", m.S());
    for (long k = 0; k < steps; ++k)
    {
      m.Advance(dt);
      std::printf("%.17g\n", m.S());
    }
    return 0;
  }
  std::fprintf(stderr, "usage: %s trace <seed> <episodes> <duration_s> <dt_s> | replay <length> <speed> <dt_s> <steps>\n",
               argv[0]);
  return 1;
}
