#ifndef WXFRAME_H
#define WXFRAME_H

#include <memory>
#include <vector>

#include <wx/wx.h>
#include <wx/timer.h>
#include <wx/event.h>

#include "IPhysicsEngine.h"
#include "Creature.h"
#include "PhysicsManager.h"

class WxFrame : public wxFrame {
public:
    WxFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);
    ~WxFrame();
    
private:
    PhysicsManager *physicsManager;
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);

    void OnStart(wxCommandEvent& event);
    void OnAdd(wxCommandEvent& event);
};

#endif
