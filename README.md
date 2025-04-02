# evolution
## Launch
### Build
1. You can `export MAKEFLAGS=-j8` before building, where 8 means amount of parallel processes `make` will use. But I'd not recommend you to set it to an amount of cpu threads you have because while building wxWidgets it'll eat your entire RAM and crave twice more... Don't do it.\
\
Due to git security policy it is just going to fail building wxWidgets. That's why you should execute `git config --global --add safe.directory $(pwd)/3rd_party/wxWidgets/src/wxWidgets_external`

    ```shell
    git clone https://github.com/Dartanjyan/evolution.git
    cd evolution
    git checkout cpp-recode -f
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    git config --global --add safe.directory $(pwd)/3rd_party/wxWidgets/src/wxWidgets_external
    make -C build
    ```
2. Launch: 
    ```shell
    ./build/evolution
    ```

## Plans
### Application
- **WIP:** Recoding on c++ with wxWidgets
### Creatures
- Add joint angle limits
- Add muscles (just like at the *Evolution by Keiwan Donyagard*)
- Add immunity for the creature to live through some generations (it'd be like you compete your grandfather or smth like that)
- Add reward/punishment system. For example, a little punishment for every muscle action, a big punishment for touching the ~~grass~~ ground with the specified body parts, and a dynamic award for the distance they have crossed. While living, creatures collect awards and punishments, what will then impact on selecting the best creatures
- Add sight parts for the creatures. It'll probably look like you just specify a part as 'sight part' and creature will ray cast in multiple directions and see enviroment. Today the environment is just a plain world and it seems to be a bad idea to use side parts because creature always knows the angle of it's every body part.
- Add a death part. If creature touches ground with this part then it dies. If all creatures in generation die then this generation just ends and create a new gen based on previous, like always, but not waiting for the timer.

