#ifndef WXAPP_H
#define WXAPP_H

#include <iostream>
#include <wx/wx.h>
#include "IGUI.h"

class WxApp : public wxApp, public IGUI{
public:
    bool OnInit() override;
    int Run() override;
};

#endif
