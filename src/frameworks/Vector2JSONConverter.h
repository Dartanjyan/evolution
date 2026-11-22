#ifndef VECTOR2JSONCONVERTER_H
#define VECTOR2JSONCONVERTER_H

#include <nlohmann/json.hpp>
#include "Vector2.h"  // твой тип вектор2

// Конвертация Vector2 <-> JSON
inline void to_json(nlohmann::json& j, const Vector2& v) {
    j = nlohmann::json{{"x", v.x}, {"y", v.y}};
}

inline void from_json(const nlohmann::json& j, Vector2& v) {
    v.x = j.at("x").get<float>();
    v.y = j.at("y").get<float>();
}

#endif