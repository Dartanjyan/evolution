#ifndef ICLI_H
#define ICLI_H

class ICLI {
public:
    virtual ~ICLI() = default;

    // Initialization
    virtual void Init() = 0;

    // This function is responsible for running the CLI interface.
    virtual int Run() = 0;

    virtual void ShowHelp() = 0;
    virtual void ShowVersion() = 0;
};

#endif