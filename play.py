"""Run the trained agent in the 3D simulation (alias of `main.py` without a subcommand).

    uv run play.py                    # best trained agent, 3D GUI + verbose terminal log
    uv run play.py --algo ppo         # dqn | reinforce | ppo | a2c | heuristic | random
    uv run play.py --no-render        # terminal only
"""

from main import main

if __name__ == "__main__":
    main()
