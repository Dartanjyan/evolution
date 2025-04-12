#ifndef WXFRAME_H
#define WXFRAME_H

#include <memory>
#include <vector>

#include <wx/wx.h>
#include <wx/timer.h>
#include <wx/event.h>

#include "IPhysicsEngine.h"
#include "Creature.h"

class WxFrame : public wxFrame {
public:
    WxFrame(const wxString &title, const wxPoint &pos, const wxSize &size);
    ~WxFrame();
    
private:
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnPaint(wxPaintEvent& event);
    void OnTimer(wxTimerEvent& event);

    void UpdateLogic();
    
    wxTimer* timer;
};

enum
{
    ID_TIMER = 2
};

#endif