#ifndef WXAPP_H
#define WXAPP_H

#include <wx/wx.h>
#include <iostream>

class WxApp : public wxApp {
public:
    bool OnInit() override;
};

#endif
