# evolution
## Launch
### Build
1. Install wxWidgets 3.2
2. ```bash
    git clone https://github.com/Dartanjyan/evolution.git
    cd evolution
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    make -C build -j$(nproc)
    ```
3. Launch: 
    ```bash
    ./build/evolution
    ```

## Plans
### Application
- **Working on:** Recoding on c++ with wxWidgets
### Creatures
- Add joint angle limits
- Add muscles (just like at the *Evolution by Keiwan Donyagard*)
- Add immunity for the creature to live through some generations (it'd be like you compete your grandfather or smth like that)
- Add reward/punishment system. For example, a little punishment for every muscle action, a big punishment for touching the ~~grass~~ ground with the specified body parts, and a dynamic award for the distance they have crossed. While living, creatures collect awards and punishments, what will then impact on selecting the best creatures
- Add sight parts for the creatures. It'll probably look like you just specify a part as 'sight part' and creature will ray cast in multiple directions and see enviroment. Today the environment is just a plain world and it seems to be a bad idea to use side parts because creature always knows the angle of it's every body part.
- Add a death part. If creature touches ground with this part then it dies. If all creatures in generation die then this generation just ends and create a new gen based on previous, like always, but not waiting for the timer.

