// WxFrame.cpp
#include "WxFrame.h"

WxFrame::WxFrame(const wxString &title, const wxPoint &pos, const wxSize &size) : wxFrame(nullptr, wxID_ANY, title, pos, size) 
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

    wxPanel *panel_top = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, 100));
    panel_top->SetBackgroundColour(wxColor(100, 100, 200));

    wxPanel *panel_bottom = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, 100));
    panel_bottom->SetBackgroundColour(wxColor(100, 200, 100));

    wxPanel *panel_bottom_right = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxSize(200, 100));
    panel_bottom_right->SetBackgroundColour(wxColor(200, 100, 100));

    wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(panel_top, 1, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer *sizer_bottom = new wxBoxSizer(wxHORIZONTAL);
    sizer_bottom->Add(panel_bottom, 1, wxEXPAND | wxALL, 10);
    sizer_bottom->Add(panel_bottom_right, 0, wxEXPAND | wxALL, 10);

    sizer->Add(sizer_bottom, 1, wxEXPAND | wxALL, 10);

    this->SetSizerAndFit(sizer);
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