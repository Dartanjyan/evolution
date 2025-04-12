#ifndef IGUI_H
#define IGUI_H

class IGUI {
public:
    virtual ~IGUI() = default;

    // Initialization
    virtual void Init() = 0;

    // This function is responsible for running the GUI interface.
    virtual int Run() = 0;
};

#endif