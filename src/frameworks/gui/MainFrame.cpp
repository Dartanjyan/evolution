#include "MainFrame.h"
#include "MainMenuPanel.h"
#include "SimulationPanel.h"
#include "SettingsPanel.h"
#include "SimulationLoadPanel.h"
#include <iostream>

MainFrame::MainFrame(PhysicsManager* physicsManager)
    : wxFrame(nullptr, wxID_ANY, "Evolution by Alex", wxDefaultPosition, wxSize(1024, 768)),
      physicsManager(physicsManager)
{
    SetSizerAndFit(new wxBoxSizer(wxVERTICAL));

    ShowMenu();
    Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent& evt) {
        wxTheApp->ExitMainLoop();
    });

    SetMinSize(wxSize(600, 480));
    // SetMinSize(wxSize(1600, 900));
}

void MainFrame::ClearCurrentPanel() {
    if (GetSizer()) {
        GetSizer()->Clear(true);
    }
}

void MainFrame::ShowSimulationLoadScreen() {
    ClearCurrentPanel();
    currentPanel = new SimulationLoadPanel(this);
    GetSizer()->Add(currentPanel, 1, wxEXPAND);
    Layout();
}

void MainFrame::ShowMenu() {
    ClearCurrentPanel();
    currentPanel = new MainMenuPanel(this);
    GetSizer()->Add(currentPanel, 1, wxEXPAND);
    Layout();
}

void MainFrame::ShowSimulation() {
    ClearCurrentPanel();
    currentPanel = new SimulationPanel(this, physicsManager);
    GetSizer()->Add(currentPanel, 1, wxEXPAND);
    Layout();
}

void MainFrame::ShowSettings() {
    ClearCurrentPanel();
    currentPanel = new SettingsPanel(this);
    GetSizer()->Add(currentPanel, 1, wxEXPAND);
    Layout();
}
