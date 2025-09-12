#include "SimulationFrame.h"
#include "DrawPanel.h"
#include <wx/dcbuffer.h>

SimulationFrame::SimulationFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos, const wxSize &size)
    : wxFrame(nullptr, wxID_ANY, title, pos, size), physicsManager(physicsManager)
{
    // Setting up a menu bar
    wxMenu *menuFile = new wxMenu;
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);

    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT, "&About\tF1", "Show about dialog");

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");
    SetMenuBar(menuBar);

    CreateStatusBar();
    SetStatusText("Simulation Running");
	
    SetMinSize(wxSize(400, 300));
    SetSize(size);
    Center();

    // Filling frame with gui stuff
    DrawPanel *drawPanel = new DrawPanel(std::move(physicsManager), this, wxID_ANY);
    drawPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

    wxPanel *controlPanel = new wxPanel(this, wxID_ANY);
    controlPanel->SetBackgroundColour(wxColour(255, 255, 255));
    // wxButton *startButton = new wxButton(controlPanel, ID_START, "Stop", wxDefaultPosition, wxSize(60, wxDefaultSize.y));
    wxButton *addButton = new wxButton(controlPanel, ID_ADD_CREATURE, "+", wxPoint(0, 0), wxSize(60, wxDefaultSize.y));

    wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(controlPanel, 0, wxEXPAND | wxBOTTOM, 1);
    sizer->Add(drawPanel, 1, wxEXPAND | wxALL);
    this->SetSizer(sizer);

    Bind(wxEVT_MENU, &SimulationFrame::OnQuit, this, wxID_EXIT);
    Bind(wxEVT_CLOSE_WINDOW, &SimulationFrame::OnCloseWindow, this, wxID_EXIT);
    Bind(wxEVT_MENU, &SimulationFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_BUTTON, &SimulationFrame::OnStart, this, ID_START);
    Bind(wxEVT_BUTTON, &SimulationFrame::OnAdd, this, ID_ADD_CREATURE);

    this->physicsManager->start();
}

SimulationFrame::~SimulationFrame()
{
}

void SimulationFrame::OnQuit(wxCommandEvent& event)
{
    HandleExit();
}

void SimulationFrame::OnCloseWindow(wxCloseEvent &event)
{
    HandleExit();
    event.Skip();
}

void SimulationFrame::HandleExit()
{
    std::cout << "Exiting application" << std::endl;
    this->physicsManager->stop();
    Close(true);
}

void SimulationFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("Physics Simulation using Chipmunk and wxWidgets",
        "About", wxOK | wxICON_INFORMATION | wxSTAY_ON_TOP | wxCENTER);
}

void SimulationFrame::OnStart(wxCommandEvent &event)
{
    this->physicsManager->stop();
}

void SimulationFrame::OnAdd(wxCommandEvent &event)
{
    for (int i=0; i<1; i++) {
        this->physicsManager->addCreature(Creature::createBasicCreature());
    }
}
