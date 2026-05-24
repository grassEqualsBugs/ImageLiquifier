#pragma once

#ifdef __OBJC__
#import <MetalKit/MetalKit.h>

@protocol InputHandler <NSObject>
- (void)mouseDown:(NSEvent *)e;
- (void)mouseUp:(NSEvent *)e;
- (void)mouseMoved:(NSEvent *)e;
- (void)mouseDragged:(NSEvent *)e;
- (void)scrollWheel:(NSEvent *)e;
- (void)keyDown:(NSEvent *)e;
- (void)keyUp:(NSEvent *)e;
@end

@interface SimView : MTKView
@end

@interface Renderer : NSObject <MTKViewDelegate, InputHandler>
- (instancetype)initWithMetalKitView:(MTKView *)view;
- (void)handleDroppedImage:(NSImage *)image;
@end
#endif
