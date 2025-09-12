#ifndef SIMULATIONFRAME_H
#define SIMULATIONFRAME_H

#include <memory>
#include <vector>

#include <wx/wx.h>
#include <wx/timer.h>
#include <wx/event.h>

#include "IPhysicsEngine.h"
#include "Creature.h"
#include "PhysicsManager.h"

class SimulationFrame : public wxFrame {
public:
    SimulationFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);
    ~SimulationFrame();
    
private:
    PhysicsManager* physicsManager;
    void OnQuit(wxCommandEvent& event);
    void OnCloseWindow(wxCloseEvent& event);
    void HandleExit();
    void OnAbout(wxCommandEvent& event);

    void OnStart(wxCommandEvent& event);
    void OnAdd(wxCommandEvent& event);
};

#endif
