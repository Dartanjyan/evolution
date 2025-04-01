// WxFrame.cpp
#include "WxFrame.h"

WxFrame::WxFrame() : wxFrame(nullptr, wxID_ANY, "Simulation") 
{
    wxMenu *menuFile = new wxMenu;
    
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);

    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");

    SetMenuBar(menuBar);

    CreateStatusBar();
    SetStatusText("Welcome to wxWidgets!");

    Bind(wxEVT_MENU, &WxFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_MENU, &WxFrame::OnExit, this, wxID_EXIT);
}

void WxFrame::OnExit(wxCommandEvent& event)
{
    Close(true);
}

void WxFrame::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("This is a wxWidgets Hello World example",
        "About Hello World", wxOK | wxICON_INFORMATION | wxSTAY_ON_TOP | wxCENTER);
}