#ifndef VECTOR2_H
#define VECTOR2_H
#include <cmath>
#include <ostream>

struct Vector2 {
    float x = 0.0f;
    float y = 0.0f;
    
    Vector2() = default;
    Vector2(float x, float y) : x(x), y(y) {}
    
    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }
    
    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }
    
    Vector2 operator*(float scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    Vector2 operator/(float scalar) const {
        return Vector2(x / scalar, y / scalar);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector2& vec) {
        os << "vec(" << vec.x << ", " << vec.y << ")";
        return os;
    }
    
    float length() const {
        return sqrt(x*x + y*y);
    }
    
    Vector2 normalized() const {
        float len = length();
        if (len > 0) {
            return Vector2(x / len, y / len);
        }
        return *this;
    }

    Vector2 rotated(float angle) const {
        float cosA = std::cos(angle);
        float sinA = std::sin(angle);
        return Vector2(
            x * cosA - y * sinA,
            x * sinA + y * cosA
        );
    }

    Vector2 revX() const {
        return Vector2(-x, y);
    }

    Vector2 revY() const {
        return Vector2(x, -y);
    }
};

#endif
