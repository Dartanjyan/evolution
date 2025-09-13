#ifndef SIMULATIONFRAME_H
#define SIMULATIONFRAME_H

#include <wx/wx.h>

#include "Frame.h"
#include "Creature.h"
#include "PhysicsManager.h"

class SimulationFrame : public Frame {
public:
    SimulationFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos = wxDefaultPosition, const wxSize &size = wxDefaultSize);
    ~SimulationFrame();
private:
    PhysicsManager* physicsManager;

    void HandleExit() override;

    void OnQuit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnAdd(wxCommandEvent& event);
};

#endif
