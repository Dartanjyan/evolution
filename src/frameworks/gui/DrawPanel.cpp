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
    drawCommandCollector = std::make_shared<DrawCommandCollector>(physicsManager);
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

void drawCircle(wxDC& dc, DrawCommand& command) {
    wxColour col = wxColour(command.color.red, command.color.green, command.color.blue, command.color.alpha);
    dc.SetBrush(col);
    setPen(dc, wxPen(wxColour(0, 0, 0), 1));
    dc.DrawCircle(command.points[0].x, command.points[0].y, command.width);
}

void drawText(wxDC& dc, DrawCommand& command) {
    dc.SetTextForeground(wxColour(command.color.red, command.color.green, command.color.blue, command.color.alpha));
    wxFont font(command.width, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    dc.SetFont(font);
    dc.DrawText(command.text, command.points[0].x, command.points[0].y);
}

void drawLine(wxDC& dc, DrawCommand& command) {
    setPen(dc, wxPen(wxColour(command.color.red, command.color.green, command.color.blue, command.color.alpha), command.width));
    dc.DrawLine(wxPoint(command.points[0].x, command.points[0].y), wxPoint(command.points[1].x, command.points[1].y));
}

void drawPolygon(wxDC& dc, DrawCommand& command) {
    dc.SetBrush(wxColour(command.color.red, command.color.green, command.color.blue, command.color.alpha));
    setPen(dc, wxPen("black"));
    std::vector<wxPoint> points;
    for (const auto& v : command.points) {
        points.emplace_back(v.x, v.y);
    }
    dc.DrawPolygon(points.size(), points.data());
}

void executeCommand(wxDC& dc, DrawCommand& command) {
    switch (command.type) {
        case DrawCommandType::TEXT:
            drawText(dc, command);
            break;
        case DrawCommandType::CIRCLE:
            drawCircle(dc, command);
            break;
        case DrawCommandType::LINE:
            drawLine(dc, command);
            break;
        case DrawCommandType::POLYGON:
            drawPolygon(dc, command);
            break;
    }
}

void DrawPanel::OnPaint(wxPaintEvent& event) {
    const wxSize size = GetClientSize();
    if (size.x <= 0 || size.y <= 0) return;

    wxBufferedPaintDC dc(this);
    dc.SetBackground(*wxWHITE);
    dc.Clear();
    setPen(dc, wxPen(wxColour("black"), 0), true);

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
    // std::cout<<"Size: "<<_size.x<<", "<<_size.y<<"\n";
    // std::cout<<"resize\n";
}

