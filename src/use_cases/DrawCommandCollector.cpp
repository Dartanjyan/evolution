#include <chrono>
#include "DrawCommandCollector.h"

#define DRAW_NEURAL_NETWORK 0

DrawCommandCollector::DrawCommandCollector(PhysicsManager *physicsManager)
: physicsManager(physicsManager)
{
    backBufferReady.store(false);
    frontBufferReady.store(false);

    physicsManager->start();
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
    physicsManager->stop();
    std::cout << "DrawCommandCollector::stop(): Stopped successfully.\n";
}

#if DRAW_NEURAL_NETWORK
template<typename T>
constexpr T map_range(T x,
                      T in_min, T in_max,
                      T out_min, T out_max)
{
    return (x - in_min) * (out_max - out_min)
           / (in_max - in_min)
           + out_min;
}
#endif

void DrawCommandCollector::updateBackBuffer()
{
    static const Color world_shape_color = Color(79, 73, 85);

    static const Color poly_color = Color(170, 153, 137);
    static const Color segment_color = Color(115, 126, 137);
    static const Color circle_color = segment_color;
    static const Color muscle_color = Color(255, 129, 110);
    static const int muscle_width = 4;

    std::vector<BodyObject> bodies {};
    std::vector<ShapeObject> shapes {};
    std::vector<ConstraintObject> constraints {};
    physicsManager->getRenderObjects(bodies, shapes, constraints);

    std::vector<const ShapeObject *> circles, segments, polygons, world_circles, world_segments, world_polygons;
    std::vector<const ConstraintObject *> constraints_objects;

    std::lock_guard<std::mutex> lock(bufferMutex);

    #if DRAW_NEURAL_NETWORK
    std::vector<Creature *> creatures;
    physicsManager->getCreatures(creatures);
    std::vector<std::size_t> layers;
    int addToReserve;
    if (creatures.size() > 0) {
        layers = creatures.at(0)->getBrain()->getLayerSizes();
    }
    for (size_t i = 0; i < layers.size(); i++) {
        addToReserve += layers[i];
        if (i > 0) {
            addToReserve += layers[i] * layers[i-1];
        }
    }
    // NOTE: Probably not correct addToReserve calculated
    #endif

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
        #if DRAW_NEURAL_NETWORK
        + addToReserve
        #endif
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
        
        std::vector<Vector2> points {Vector2(0, 0), Vector2(0, 0)};
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
    // Segments outline
    for (const auto *shape : segments) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        
        std::vector<Vector2> points {Vector2(0, 0), Vector2(0, 0)};
        for (int i = 0; i < 2; ++i) {
            points[i] = shape->vertices[i].rotated(angle) + body->position;
        }

        backBuffer->emplace_back(DrawCommandType::LINE, Color(0, 0, 0), points, radius-1);
    }
    // Segments
    for (const auto *shape : segments) {
        const BodyObject* body = shape->body;
        const float angle = body->angle;
        const float radius = shape->radius;
        
        std::vector<Vector2> points {Vector2(0, 0), Vector2(0, 0)};
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

    DrawCommand command(DrawCommandType::TEXT, Color(100, 100, 100), std::vector<Vector2>{Vector2(10, 10)}, 12);  // 12 is font size
    
    char buf[256];
    snprintf(buf, sizeof(buf), "FPS: %.1f\nGeneration: %u", fps, physicsManager->getGeneration());
    command.text = buf;
    backBuffer->emplace_back(command);

    // TODO: Move to another function and variable in order to not render it every frame
    // Neural network of the first creature
    #if DRAW_NEURAL_NETWORK
    if (creatures.size() > 4) {
        Brain *brain = creatures[creatures.size()-3]->getBrain();
        std::vector<size_t> layers = brain->getLayerSizes();
        std::vector<std::vector<double>> weights = brain->getWeights();

        double minWeight, maxWeight;
        bool weightsInitialized = false;
        for (auto vec : weights) {
            for (auto w : vec) {
                if (!weightsInitialized) {
                    minWeight = w;
                    maxWeight = w;
                    weightsInitialized = true;
                }
                if (w > maxWeight) {
                    maxWeight = w;
                } else if (w < minWeight) {
                    minWeight = w;
                }
            }
        }

        const Vector2 panel = panelSize.load();
        const int minX = 0;
        const int maxX = panel.x;
        const int minY = 360;
        const int maxY = panel.y;
        const int stepX = (maxX - minX) / (layers.size() + 3);
        
        std::vector<Vector2> previousPositions1;
        std::vector<Vector2> previousPositions2;
        std::vector<Vector2>& previousPositionsFront = previousPositions1;
        std::vector<Vector2>& previousPositionsBack = previousPositions2;
        {
            // Draw the first layer
            const int posX = stepX * 1 + minX;
            const int stepY = (maxY - minY) / (layers[0] + 2);
            for (size_t j = 0; j < layers[0]; ++j) {
                const int posY = stepY * (j+1) + minY;
                backBuffer->emplace_back(DrawCommandType::CIRCLE, Color(178, 75, 23), Vector2(posX, posY), 5);
                previousPositionsFront.emplace_back(posX, posY);
            }
        }

        for (size_t l = 0; l + 1 < layers.size(); ++l) {
            // For every layer except the least

            const int posX = stepX * (l+2) + minX;
            const int stepY = (maxY - minY) / (layers[l + 1] + 2);
            for (size_t j = 0; j < layers[l + 1]; ++j) {
                // For every neuron on layer

                const int posY = stepY * (j+1) + minY;
                
                for (size_t i = 0; i < layers[l]; ++i) {
                    // For every weight
                    size_t idx = j * layers[l] + i;
                    const double minWidth = 0.1;
                    const double maxWidth = 3;

                    for (const auto v : previousPositionsFront) {
                        std::vector<Vector2> points { Vector2(posX, posY), v };
                        uint8_t gray = map_range(weights[l][idx], minWeight, maxWeight, 0.0, 255.0);
                        backBuffer->emplace_back(DrawCommandType::LINE, Color(gray, gray, gray, gray), points, map_range(weights[l][idx], minWeight, maxWeight, minWidth, maxWidth));
                    }
                }
                backBuffer->emplace_back(DrawCommandType::CIRCLE, Color(178, 75, 23), Vector2(posX, posY), 5);
                previousPositionsBack.emplace_back(posX, posY);
            }
            auto& tmp = previousPositionsFront;
            previousPositionsFront = previousPositionsBack;
            previousPositionsBack = tmp;
            previousPositionsBack.clear();
        }
    }
    #endif
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
