#ifndef WXAPP_H
#define WXAPP_H

#include <wx/wx.h>

class WxApp : public wxApp {
public:
    bool OnInit() override;
};

#endif
