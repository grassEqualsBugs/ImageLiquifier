#pragma once

#ifdef __METAL_VERSION__
    #include <metal_stdlib>
    using namespace metal;
    typedef float2 SimFloat2;
#else
    #include <simd/simd.h>
    #include <stdint.h>
    typedef simd_float2 SimFloat2;
#endif

typedef struct {
    float cellSize;
    float deltaTime;
    float fluidDensity;
    int width;
    int height;
    float mouseRadius;
    float velocityStrength;
    float weightSOR;
} SimConstants;

typedef struct {
    SimFloat2 pos;
    SimFloat2 delta;
    bool leftDown;
    bool rightDown;
    bool isSolidMode;
    bool isPaused;
} MouseState;

typedef struct {
    MouseState mouse;
} FrameData;
