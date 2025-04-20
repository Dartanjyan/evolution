#include "DrawPanel.h"
#include <wx/dcbuffer.h>
#include <chrono>

DrawPanel::DrawPanel(PhysicsManager* physicsManager, wxWindow* parent, wxWindowID id,
                     const wxPoint& pos,
                     const wxSize& size,
                     long style) 
    : physicsManager(physicsManager), wxPanel(parent, id, pos, size, style) {
    
    if (!physicsManager) {
        std::cout << "DrawPanel constructor: got nullptr as physicsManager\n";
    }
        
    SetBackgroundColour(wxColour(240, 240, 240));
    
    timer = new wxTimer(this, ID_TIMER);
    timer->SetOwner(this, ID_TIMER);
    timer->Start(16);
    
    
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

/*
void DrawPanel::OnPaint(wxPaintEvent& event) {
    wxBufferedPaintDC dc(this);
    
    // Очистка фона (использует цвет фона панели)
    dc.Clear();
    
    // Рисуем только в пределах этой панели
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.SetPen(*wxBLACK_PEN);
    dc.DrawRectangle(m_positionX, m_positionY, 50, 50);
    
    // Можно добавить границу для наглядности
    dc.SetPen(*wxRED_PEN);
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.DrawRectangle(0, 0, GetSize().x, GetSize().y);
}
*/

void DrawPanel::OnPaint(wxPaintEvent& event) {
    wxBufferedPaintDC dc(this);

    dc.SetBackground(*wxWHITE);
    // Рисуем фон
    dc.Clear();

    std::vector<BodyObject> bodies {};
    std::vector<ShapeObject> shapes {};
    std::vector<ConstraintObject> constraints {};
    physicsManager->getRenderObjects(bodies, shapes, constraints);
    // Отрисовка ShapeObject
    for (const auto& shape : shapes) {
        if (!shape.body) {
            std::cout << "Shape with id="<<shape.id<<" has no body\n";
            continue;
        }

        const BodyObject* body = shape.body;

        if(shape.vertices.size() == 2) {
            // Если 2 вершины, рисуем линию (сегмент)
            wxPoint start(
                (shape.vertices[0].rotated(body->angle) + body->position).x,
                (shape.vertices[0].rotated(body->angle) + body->position).y
            );
            wxPoint end(
                (shape.vertices[1].rotated(body->angle) + body->position).x,
                (shape.vertices[1].rotated(body->angle) + body->position).y
            );
            dc.SetPen(wxPen(*wxBLACK, 7/2)); // Толщина линии = radius
            dc.DrawLine(start, end);
	    } else {
            // Если 3 и больше вершин, рисуем многогранник
            std::vector<wxPoint> points;
            for (const auto& vertex : shape.vertices) {
                Vector2 rotated = vertex.rotated(body->angle);
                points.emplace_back(
                    rotated.x + body->position.x,
                    rotated.y + body->position.y
                );
            }
            dc.SetBrush(*wxBLUE_BRUSH);
            dc.SetPen(*wxBLACK_PEN);
            dc.DrawPolygon(points.size(), points.data());
        }
    }

    
    // Отрисовка BodyObject
    dc.SetBrush(*wxCYAN_BRUSH);
    dc.SetPen(*wxBLACK_PEN);
    for (const auto& body : bodies) {
        dc.DrawCircle(wxPoint(body.position.x, body.position.y), 10);
    }
    

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

void DrawPanel::OnTimer(wxTimerEvent& event) {
    Refresh(false);
    Update();
}

void DrawPanel::OnSize(wxSizeEvent& event) {
    Refresh(); // Перерисовываем при изменении размера
    event.Skip(); // Пропускаем событие дальше
}

