#include "MainMenuFrame.h"
#include "SimulationFrame.h"
#include "SettingsFrame.h"
#include "WxIds.h"

MainMenuFrame::MainMenuFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos, const wxSize &size)
    : wxFrame(nullptr, wxID_ANY, title, pos, size), physicsManager(physicsManager) {
    
    wxPanel* panel = new wxPanel(this);
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    
    int margin = 15;
    int padding = 10;
    wxGridSizer* gridSizer = new wxGridSizer(2, 2, padding, padding);
    
    // Menu buttons
    wxButton* simulationBtn = new wxButton(panel, ID_MENU_SIMULATION, "Simulation");
    wxButton* settingsBtn = new wxButton(panel, ID_MENU_SETTINGS, "Settings");
    wxButton* exitBtn = new wxButton(panel, wxID_EXIT, "Exit");
    

    // sizer->Add(simulationBtn, 0, sizerFlags, margin);
    // sizer->Add(settingsBtn, 0, sizerFlags, margin);
    // sizer->Add(exitBtn, 0, sizerFlags, margin);
    
    gridSizer->Add(simulationBtn, 0, wxLEFT | wxUP | wxEXPAND, margin);
    gridSizer->AddSpacer(0);
    gridSizer->Add(settingsBtn, 0, wxLEFT | wxDOWN | wxEXPAND, margin);
    gridSizer->Add(exitBtn, 0, wxRIGHT | wxDOWN | wxEXPAND, margin);
    
    panel->SetSizer(gridSizer);
    SetClientSize(250, 150);
    SetMinClientSize(wxSize(250, 150));
    Center();
    
    Bind(wxEVT_BUTTON, &MainMenuFrame::OnSimulation,this, ID_MENU_SIMULATION);
    Bind(wxEVT_BUTTON, &MainMenuFrame::OnSettings, this, ID_MENU_SETTINGS);
    Bind(wxEVT_BUTTON, &MainMenuFrame::OnExit, this, wxID_EXIT);
}

void MainMenuFrame::OnSimulation(wxCommandEvent& event) {
    std::cout << "Simulation clicked!\n";
    SimulationFrame* simulationFrame = new SimulationFrame(physicsManager, "Simulation");
    simulationFrame->Show(true);
    this->Hide(); // Hide menu
}

void MainMenuFrame::OnSettings(wxCommandEvent& event) {
    SettingsFrame* settingsFrame = new SettingsFrame();
    settingsFrame->Show(true);
    this->Hide(); // Hide menu
}

void MainMenuFrame::OnExit(wxCommandEvent& event) {
    Close(true);
}
