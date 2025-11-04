#ifndef CAMERA_H
#define CAMERA_H

#include "Vector2.h"

/*
Камера должна уметь:
1. Следить за определенным существом
2. Не следить за существом (тут кстати тоже вопрос, стоит ли хранить в камере лишь id отслеживаемого существа или указатель, Но мне что-то подсказывает что лучше id)
3. Иметь функционал масштаба

Насколько я понимаю DrawPanel должен иметь unique_ptr на камеру, drawcommandcollector должен также учитывать расположение камеры и область которую она охватывает
*/

class Camera {
private:
    float scale;
    unsigned trackingCreatureId;
    Vector2 pos;
public:
    Camera(): scale(1), pos(Vector2(0, 0)) {}
    void setScale(float s) { scale = s;}
};

#endif
