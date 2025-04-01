#ifndef WXFRAME_H
#define WXFRAME_H

#include <wx/wx.h>

class WxFrame : public wxFrame {
public:
    WxFrame();
private:
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
};

enum
{
    
};

#endif
