#ifndef IUI_H
#define IUI_H

class IUI {
public:
    virtual ~IUI() = default;
    virtual int Run(bool gui) = 0;
};

#endif
