#include "WxFrame.h"
#include "ChipmunkPhysicsEngine.h"
#include <wx/dcclient.h>
#include <chrono>
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
    wxPanel *panel = new wxPanel(this, wxID_ANY);
    //panel->SetBackgroundColour(wxColour(225, 220, 220));

    timer = new wxTimer(this, ID_TIMER);
    timer->SetOwner(this, ID_TIMER);
    timer->Start(16);

    Bind(wxEVT_MENU, &WxFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &WxFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_PAINT, &WxFrame::OnPaint, this);
    Bind(wxEVT_TIMER, &WxFrame::OnTimer, this, ID_TIMER);
}

void WxFrame::OnPaint(wxPaintEvent& event) {
    wxBufferedPaintDC dc(this);
    
    dc.SetBackground(*wxWHITE);
    // Рисуем фон
    dc.Clear();
    // Получаем текущее время
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&now_time);
    time_str.pop_back(); // Удаляем символ новой строки
    // Устанавливаем цвет текста
    dc.SetTextForeground(*wxBLACK);
    // Устанавливаем шрифт
    wxFont font(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    dc.SetFont(font);
    // Выводим текст на экран
    dc.DrawText("Current time: " + wxString(time_str), 10, 30);

    // FPS counter
    static wxStopWatch sw;
    static int frameCount = 0;
    static float fps = 0;
    
    frameCount++;
    if (sw.Time() > 500) {
        fps = frameCount / (sw.Time() / 1000.0f);
        frameCount = 0;
        sw.Start();
    }
    
    dc.DrawText(wxString::Format("FPS: %.1f", fps), 10, 10);
}

void WxFrame::OnTimer(wxTimerEvent& event) {
    Refresh(false);
    Update();
}


WxFrame::~WxFrame()
{
    if (timer && false) {
        timer->Stop();
        delete timer;
        timer = nullptr;
    }
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

