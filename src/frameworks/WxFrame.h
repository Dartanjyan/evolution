#ifndef WXFRAME_H
#define WXFRAME_H

#include <wx/wx.h>
#include <wx/timer.h>
#include "IPhysicsEngine.h"
#include "Creature.h"
#include <memory>
#include <vector>

class WxFrame : public wxFrame {
public:
    WxFrame(const wxString &title, const wxPoint &pos, const wxSize &size);
    ~WxFrame();
    
private:
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnPaint(wxPaintEvent& event);
    void OnTimer(wxTimerEvent& event);
    
    std::unique_ptr<IPhysicsEngine> physicsEngine;
    std::vector<Creature*> creatures;
    
    wxTimer* timer;
    
    void InitializeSimulation();
    void UpdateSimulation(float deltaTime);
    void CleanupSimulation();
    
    void CreateTestCreature();
    void RenderCreatures(wxDC& dc);
    
    wxDECLARE_EVENT_TABLE();
};

enum
{
    ID_TIMER = 1
};

#endif