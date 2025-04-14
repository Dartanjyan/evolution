#include "WxFrame.h"
#include "DrawPanel.h"
#include <wx/dcclient.h>
#include <wx/dcbuffer.h>

WxFrame::WxFrame(PhysicsManager* physicsManager, const wxString &title, const wxPoint &pos, const wxSize &size)
    : wxFrame(nullptr, wxID_ANY, title, pos, size)
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
	
    SetMinSize(wxSize(400, 200));
    SetClientSize(wxSize(800, 600));
    Center();

    // Filling frame with gui stuff
    DrawPanel *drawPanel = new DrawPanel(physicsManager, this, wxID_ANY);
    drawPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);

    wxPanel *controlPanel = new wxPanel(this, wxID_ANY);
    controlPanel->SetBackgroundColour(wxColour(255, 255, 255));
    wxButton *startButton = new wxButton(controlPanel, ID_START, "Start", wxDefaultPosition, wxDefaultSize);

    wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(controlPanel, 0, wxEXPAND | wxBOTTOM, 1);
    sizer->Add(drawPanel, 1, wxEXPAND | wxALL);
    this->SetSizer(sizer);

    Bind(wxEVT_MENU, &WxFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &WxFrame::OnAbout, this, wxID_ABOUT);
}

WxFrame::~WxFrame()
{
}

void WxFrame::OnExit(wxCommandEvent& event)
{
    Close(true);
    std::cout << "Exiting application" << std::endl;
}

void WxFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("Physics Simulation using Chipmunk and wxWidgets",
        "About", wxOK | wxICON_INFORMATION | wxSTAY_ON_TOP | wxCENTER);
}

