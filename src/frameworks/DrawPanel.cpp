#include "DrawPanel.h"

#include <wx/dcbuffer.h>
#include <chrono>
#include <thread>
#include <array>

DrawPanel::DrawPanel(PhysicsManager* physicsManager, wxWindow* parent, wxWindowID id,
                     const wxPoint& pos,
                     const wxSize& size,
                     long style) 
    : wxPanel(parent, id, pos, size, style), physicsManager(physicsManager)
{
    drawCommandCollector = std::make_unique<DrawCommandCollector>(physicsManager);
    drawCommandCollector->start();
        
    SetBackgroundColour(wxColour(240, 240, 240));
    
    timer = new wxTimer(this, ID_TIMER);
    timer->SetOwner(this, ID_TIMER);
    timer->Start(1000/60);
    
    Bind(wxEVT_SIZE, &DrawPanel::OnSize, this);
    Bind(wxEVT_PAINT, &DrawPanel::OnPaint, this);
    Bind(wxEVT_TIMER, &DrawPanel::OnTimer, this, ID_TIMER);
}

DrawPanel::~DrawPanel()
{
    if (timer) {
        timer->Stop();
        delete timer;
        timer = nullptr;
    }
}

void setPen(wxDC &dc, wxPen pen, bool force = false) {
    static wxPen last_pen;

    if (force || last_pen != pen) {
        dc.SetPen(pen);
        last_pen = pen;
    }
}

void drawText(wxDC& dc, DrawCommand& command) {
    dc.SetTextForeground(*wxColor(command.color.red, command.color.green, command.color.blue, command.color.alpha));
    wxFont font(command.width, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    dc.SetFont(font);
    dc.DrawText(command.text, command.points[0].x, command.points[0].y);
}

void drawLine(wxDC& dc, DrawCommand& command) {
    setPen(dc, wxPen(wxColour(command.color.red, command.color.green, command.color.blue, command.color.alpha), command.width));
    dc.DrawLine(wxPoint(command.points[0].x, command.points[0].y), wxPoint(command.points[1].x, command.points[1].y));
}

void executeCommand(wxDC& dc, DrawCommand& command) {
    switch (command.type) {
        case DrawCommandType::TEXT:
            drawText(dc, command);
            break;
        case DrawCommandType::CIRCLE:
            break;
        case DrawCommandType::LINE:
            drawLine(dc, command);
            break;
        case DrawCommandType::POLYGON:
            break;
    }
}

void DrawPanel::OnPaint(wxPaintEvent& event) {
    const wxSize size = GetClientSize();
    if (size.x <= 0 || size.y <= 0) return;

    wxBufferedPaintDC dc(this);
    dc.SetBackground(*wxWHITE);
    dc.Clear();

    std::vector<DrawCommand> commands;
    drawCommandCollector->getCommands(commands);
    
    for (auto& command : commands) {
        executeCommand(dc, command);
    }
}

void DrawPanel::OnTimer(wxTimerEvent& event) {
    Refresh(false);
    Update();
}

void DrawPanel::OnSize(wxSizeEvent& event) {
    Refresh();
    event.Skip();
    
    auto _size = GetClientSize();
    drawCommandCollector->setPanelSize(Vector2(_size.x, _size.y));
    std::cout<<"Size: "<<_size.x<<", "<<_size.y<<"\n";
    // std::cout<<"resize\n";
}

