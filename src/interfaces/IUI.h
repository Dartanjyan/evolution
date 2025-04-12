#ifndef IUI_H
#define IUI_H

// #include "IGUI.h"
// #include "ICLI.h"

class IUI {
public:
    virtual ~IUI() = default;
    virtual int Run(bool gui) = 0;
    virtual void Init(bool gui) = 0;
};

#endif
