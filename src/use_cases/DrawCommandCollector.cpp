#include <chrono>
#include "DrawCommandCollector.h"

DrawCommandCollector::DrawCommandCollector(PhysicsManager *physicsManager)
: physicsManager(physicsManager)
{
    backBufferReady.store(false);
    frontBufferReady.store(false);
}

DrawCommandCollector::~DrawCommandCollector()
{
    stop();
}

void DrawCommandCollector::getCommands(std::vector<DrawCommand> &commands)
{
    bufferMutex.lock();
    commands = *frontBuffer;
    bufferMutex.unlock();
    
    if (backBufferReady.load()) {
        flip();
    }
}

void DrawCommandCollector::setPanelSize(Vector2 newSize)
{
    panelSize.store(newSize);
    std::lock_guard<std::mutex> lock(bufferMutex);
    // NOTE: not sure if this will work
    backBufferReady.store(false);
}

void DrawCommandCollector::flip()
{
    std::lock_guard<std::mutex> lock(bufferMutex);

    // Swap frames
    auto* tmp = frontBuffer;
    frontBuffer = backBuffer;
    backBuffer = tmp;
    
    // Request render
    backBufferReady.store(false);
}

void DrawCommandCollector::stop() {
    running.store(false);
    if (collectorThread.joinable())
        collectorThread.join();
    std::cout << "DrawCommandCollector::stop(): Stopped successfully.\n";
}

void DrawCommandCollector::updateBackBuffer()
{
    const Color world_shape_color = Color(79, 73, 85);

    Color poly_color = Color(170, 153, 137);
    Color segment_color = Color(115, 126, 137);
    Color circle_color = segment_color;
    Color muscle_color = Color(255, 129, 110);
    const int muscle_width = 4;

    std::vector<BodyObject> bodies {};
    std::vector<ShapeObject> shapes {};
    std::vector<ConstraintObject> constraints {};
    physicsManager->getRenderObjects(bodies, shapes, constraints);

    std::vector<const ShapeObject*> circles, segments, polygons, world_circles, world_segments, world_polygons;
    std::vector<const ConstraintObject*> constraints_objects;

    std::lock_guard<std::mutex> lock(bufferMutex);

    // Fill vectors do draw them with different colors.
    for (auto& s: shapes) {
        if (s.isWorldObj) {
            switch (s.shapeType) {
                case ShapeType::Circle:   world_circles.push_back(&s); break;
                case ShapeType::Segment:  world_segments.push_back(&s); break;
                case ShapeType::Polygon:  world_polygons.push_back(&s); break;
            }
        } else {
            switch (s.shapeType) {
                case ShapeType::Circle:   circles.push_back(&s); break;
                case ShapeType::Segment:  segments.push_back(&s); break;
                case ShapeType::Polygon:  polygons.push_back(&s); break;
            }
        }

        if (!s.body) {
            std::cout<<"Shape with id="<<s.id<<" has no body\n";
            continue;
        }
    }
    for (auto& c: constraints) {
        switch (c.constraintType) {
            case ConstraintType::MUSCLE:
                constraints_objects.push_back(&c);
                break;
            default: break;
        }
    }


    // ===========Drawing=============
    
    backBuffer->clear();
    backBuffer->reserve(
        circles.size() + segments.size() + polygons.size()
        + world_circles.size() + world_segments.size() + world_polygons.size()
        + constraints_objects.size()
        + 1     // FPS text
    );

    for (const auto *shape : world_polygons) {
        const BodyObject* body = shape->body;

        std::vector<Vector2> points;
        for (const auto& v : shape->vertices) {
            Vector2 vertex = v.rotated(body->angle) + body->position;
            points.emplace_back(vertex.x, vertex.y);
        }
        
        backBuffer->emplace_back(DrawCommandType::POLYGON, Color(0, 0, 0), points, shape->radius);
    }

    for (const auto *shape : world_segments) {
        const BodyObject* body = shape->body;
        
        std::vector<Vector2> points;
        for (int i = 0; i < 2; ++i) {
            points[i] = shape->vertices[i].rotated(body->angle) + body->position;
        }

        backBuffer->emplace_back(DrawCommandType::LINE, world_shape_color, points, shape->radius - 1);
    }

    for (const auto *shape : world_circles) {
        const BodyObject* body = shape->body;
        backBuffer->emplace_back(DrawCommandType::CIRCLE, world_shape_color, shape->vertices[0].rotated(body->angle) + body->position, shape->radius);
    }

    // First draw constraints
    for (const auto *constraint : constraints_objects) {
        const BodyObject* partA = constraint->partA;
        const BodyObject* partB = constraint->partB;
        if (!partA || !partB) {
            std::cout << "Constraint with id=" << constraint->id << " has no partA or partB\n";
            continue;
        }
        const Vector2 anchorA = constraint->anchorA + partA->position;
        const Vector2 anchorB = constraint->anchorB + partB->position;

        backBuffer->emplace_back(DrawCommandType::LINE, Color(0, 0, 0), std::vector<Vector2>{anchorA, anchorB}, muscle_width);
    }
    for (const auto *constraint : constraints_objects) {
        const BodyObject* partA = constraint->partA;
        const BodyObject* partB = constraint->partB;
        if (!partA || !partB) {
            std::cout << "Constraint with id=" << constraint->id << " has no partA or partB\n";
            continue;
        }
        const Vector2 anchorA = constraint->anchorA + partA->position;
        const Vector2 anchorB = constraint->anchorB + partB->position;

        backBuffer->emplace_back(DrawCommandType::LINE, muscle_color, std::vector<Vector2>{anchorA, anchorB}, muscle_width-2);
    }

    // Second draw polygons in order for segments to be on top
    if (polygons.size() > 0) {
        for (const auto *shape : polygons) {
            const BodyObject* body = shape->body;
            const float angle = body->angle;
            const float radius = shape->radius;
    
            std::vector<Vector2> points;
            points.reserve(shape->vertices.size());
            for (const auto& v : shape->vertices) {
                points.emplace_back(v.rotated(angle) + body->position);
            }
            
            backBuffer->emplace_back(DrawCommandType::POLYGON, poly_color, points);
        }
    }
    // Segments
    for (const auto *shape : segments) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        
        std::vector<Vector2> points;
        points.reserve(2);
        for (int i = 0; i < 2; ++i) {
            points[i] = shape->vertices[i].rotated(angle) + body->position;
        }

        backBuffer->emplace_back(DrawCommandType::LINE, Color(0, 0, 0), points, radius-1);
    }
    for (const auto *shape : segments) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        
        std::vector<Vector2> points;
        points.reserve(2);
        for (int i = 0; i < 2; ++i) {
            points[i] = shape->vertices[i].rotated(angle) + body->position;
        }

        backBuffer->emplace_back(DrawCommandType::LINE, segment_color, points, radius-2);
    }

    // Circles
    for (const auto *shape : circles) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        backBuffer->emplace_back(DrawCommandType::CIRCLE, circle_color, std::vector<Vector2>{shape->vertices[0].rotated(angle) + body->position}, radius);
    }
    
    // FPS counter
    
    static auto lastTime = std::chrono::_V2::high_resolution_clock::now();
    static int frameCount = 0;
    static float fps = 0;

    frameCount++;
    const auto now = std::chrono::_V2::high_resolution_clock::now();
    const auto time = std::chrono::milliseconds((now - lastTime).count()/1000000);
    if (time > std::chrono::milliseconds(500)) {
        fps = frameCount / ((float)time.count() / 1000.0f);
        frameCount = 0;
        lastTime = now;
    }

    struct DrawCommand command(DrawCommandType::TEXT, Color(100, 100, 100), std::vector<Vector2>{Vector2(10, 10)}, 12);  // 12 is font size
    
    char buf[20];
    snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
    command.text = buf;
    backBuffer->emplace_back(command);
}

void DrawCommandCollector::start()
{
    if (running.load()) {
        std::cout << "DrawCommandCollector::start(): Already running.\n";
        return;
    }
    running.store(true);
    collectorThread = std::thread(&DrawCommandCollector::run, this);
    std::cout << "DrawCommandCollector::start(): Started successfully.\n";
}

void DrawCommandCollector::run()
{
    // Initialize front buffer
    updateBackBuffer();
    flip();
    frontBufferReady.store(true);

    while (running.load()) {
        if (!backBufferReady.load()) {
            updateBackBuffer();
            backBufferReady.store(true);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
