#include "WxFrame.h"
#include "DrawPanel.h"
#include <wx/dcclient.h>
#include <wx/dcbuffer.h>

WxFrame::WxFrame(const wxString &title, const wxPoint &pos, const wxSize &size)
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
    DrawPanel *panel = new DrawPanel(this, wxID_ANY);
    panel->SetBackgroundStyle(wxBG_STYLE_PAINT);

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

