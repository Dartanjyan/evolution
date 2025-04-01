// IRenderer.h
#ifndef IRENDERER_H
#define IRENDERER_H

class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual void Render() = 0;
};

#endif
