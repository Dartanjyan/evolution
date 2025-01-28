# evolution
## Launch
### How to
0. Install python 3.12
1. Install libraries\
   `pip install -r requirements.txt`
2. Launch main.py\
   `python main.py`
### Flags
- `--mut-alg <0-2>`: specify mutation algorithm. 2 is default
- `--load`: takes no arguments. If the flag is given, autosave will be loaded automatically at the start.

## Plans
### Creatures
- Add joint angle limits
- Add muscles (just like at the Evolution by Keiwan Donyagard)
- Add immunity for the creature to live through some generations (it'd be like you compete your grandfather or stuff like that)
- Add reward/punishment system. For example, a little punishment for every muscle action, a big punishment for touching the ~~grass~~ ground with the specified body parts, and a dynamic award for the distance they have crossed
### Application
- Add GUI: main menu, load menu, application and simulation settings, etc.
- Add GUI editor to create, save and load creatures or at least readable text way to save creatures anywhere but not hard-coded
- Add zoom
- Optimize calculating: use c++ libraries, effectively use threading, use GPU for faster AI computing
- Eventually refactor the whole `main.py`
