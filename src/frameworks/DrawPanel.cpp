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
            std::cout<<"Shape with id="<<shape.id<<" has no body\n";
            continue;
        }

        const BodyObject* body = shape.body;
        const float angle = body->angle;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float radius = shape.radius;

        switch (shape.vertices.size()) {
            case 0:
                throw std::runtime_error("Shape with id=" + std::to_string(shape.id) + " has 0 vertices");
                
            case 1: {
                const auto& v = shape.vertices[0];
                float x = v.x * cos_a - v.y * sin_a + body->position.x;
                float y = v.x * sin_a + v.y * cos_a + body->position.y;
                dc.SetBrush(*wxBLUE_BRUSH);
                dc.DrawCircle(wxPoint(x, y), radius);
                break;
            }
            
            case 2: {
                // Рисуем линию с кругами на концах
                std::array<wxPoint, 2> points;
                for (int i = 0; i < 2; ++i) {
                    const auto& v = shape.vertices[i].rotated(angle) + body->position;
                    points[i] = wxPoint(v.x, v.y);
                }
                
                // Толстая линия
                dc.SetPen(wxPen(*wxBLACK, radius - 1));
                dc.DrawLine(points[0], points[1]);
                
                /*
                // Круги на концах
                dc.SetPen(wxPen(*wxBLACK, 1));
                dc.SetBrush(*wxBLUE_BRUSH);
                dc.DrawCircle(points[0], radius-1);
                dc.DrawCircle(points[1], radius-1);
                break;
                */
            }
            
            default: {
                // Рисуем многоугольник
                std::vector<wxPoint> points;
                for (const auto& v : shape.vertices) {
                    Vector2 vertex = v.rotated(angle) + body->position;
                    points.emplace_back(vertex.x, vertex.y);
                }
                
                dc.SetBrush(*wxBLUE_BRUSH);
                dc.SetPen(*wxBLACK_PEN);
                dc.DrawPolygon(points.size(), points.data());
                break;
            }
        }
        
        /*
        if (shape.id == 228) {
            for (const auto& v : shape.vertices) {
                std::cout<<"v:\t"<<v<<"\nang:\t"<<angle<<"\nv.rot:\t"<<v.rotated(angle)<<"\nbpos:\t"<<body->position<<"\n";
            }
            std::cout<<std::endl;
        }
        */
    }

    /*
    // Отрисовка BodyObject
    dc.SetBrush(*wxCYAN_BRUSH);
    dc.SetPen(*wxBLACK_PEN);
    
    for (const auto& body : bodies) {
        dc.DrawCircle(wxPoint(body.position.x, body.position.y), 10);
    }
    */

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

