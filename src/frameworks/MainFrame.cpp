#include "MainFrame.h"
#include "MainMenuPanel.h"
#include "SimulationPanel.h"
#include "SettingsPanel.h"
#include <iostream>

MainFrame::MainFrame(PhysicsManager* physicsManager)
    : wxFrame(nullptr, wxID_ANY, "Evolution by Alex", wxDefaultPosition, wxSize(800, 600)),
      physicsManager(physicsManager)
{
    ShowMenu();
    Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent& evt) {
        wxTheApp->ExitMainLoop();
    });

    SetMinSize(wxSize(600, 480));
}

void MainFrame::ClearCurrentPanel() {
    if (currentPanel) {
        currentPanel->Destroy();
        currentPanel = nullptr;
    }
}

void MainFrame::ShowMenu() {
    ClearCurrentPanel();
    currentPanel = new MainMenuPanel(this);
    SetSizerAndFit(new wxBoxSizer(wxVERTICAL));
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
