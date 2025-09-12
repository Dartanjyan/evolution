#ifndef MAIN_MENU_FRAME_H
#define MAIN_MENU_FRAME_H

#include <wx/wx.h>
#include "PhysicsManager.h"

class MainMenuFrame : public wxFrame {
public:
    MainMenuFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);
    
private:
    PhysicsManager* physicsManager;
    
    void OnSimulation(wxCommandEvent& event);
    void OnSettings(wxCommandEvent& event);
    void OnExit(wxCommandEvent& event);
};

#endif
