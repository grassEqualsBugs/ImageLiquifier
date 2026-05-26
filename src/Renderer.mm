#import "../include/Renderer.hpp"
#import <AppKit/AppKit.h>
#import "../include/SimParams.h"
#import "../include/FluidSim.hpp"
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

@implementation SimView
- (instancetype)initWithFrame:(CGRect)frame device:(id<MTLDevice>)device {
    self = [super initWithFrame:frame device:device];
    if (self) {
        [self registerForDraggedTypes:@[NSPasteboardTypeFileURL]];
    }
    return self;
}

- (BOOL)acceptsFirstResponder { return YES; }

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    return NSDragOperationCopy;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    NSPasteboard *pboard = [sender draggingPasteboard];
    if ([[pboard types] containsObject:NSPasteboardTypeFileURL]) {
        NSArray *urls = [pboard readObjectsForClasses:@[[NSURL class]] options:nil];
        if (urls.count > 0) {
            NSURL *url = urls[0];
            NSImage *image = [[NSImage alloc] initWithContentsOfURL:url];
            if (image) {
                if ([self.delegate respondsToSelector:@selector(handleDroppedImage:)]) {
                    [(Renderer *)self.delegate handleDroppedImage:image];
                    return YES;
                }
            }
        }
    }
    return NO;
}

- (void)mouseDown:(NSEvent *)e         { [(id<InputHandler>)self.delegate mouseDown:e]; }
- (void)mouseUp:(NSEvent *)e           { [(id<InputHandler>)self.delegate mouseUp:e]; }
- (void)mouseMoved:(NSEvent *)e        { [(id<InputHandler>)self.delegate mouseMoved:e]; }
- (void)mouseDragged:(NSEvent *)e      { [(id<InputHandler>)self.delegate mouseDragged:e]; }
- (void)scrollWheel:(NSEvent *)e       { [(id<InputHandler>)self.delegate scrollWheel:e]; }
- (void)keyDown:(NSEvent *)e           { [(id<InputHandler>)self.delegate keyDown:e]; }
- (void)keyUp:(NSEvent *)e             { [(id<InputHandler>)self.delegate keyUp:e]; }
@end

// Actual implementation of renderer
@implementation Renderer {
    id<MTLBuffer> _frameDataBuffer;
   	simd_float2   _mousePos;
    simd_float2   _lastMousePos;
    bool          _firstFrameMouse;
    bool          _leftDown;
    bool          _isSolidMode;
    bool          _paused;

    NSTextField  *_fpsLabel;
    NSTimeInterval _lastFrameTime;
    double         _smoothedFPS;

	id<MTLDevice> _device;
	int           _width;
	int           _height;

	id<MTLCommandQueue>         _commandQueue;
	id<MTLRenderPipelineState>  _renderPipeline;

	FluidSim*                   _sim;
}
// Constructor (initialization of renderer)
- (instancetype)initWithMetalKitView:(MTKView *)view {
  	self = [super init];
    if (!self) return nil;

    _width = 1600;
    _height = 900;
    float simScale = 1.5f;
    int simWidth = _width / simScale;
    int simHeight = _height / simScale;

    _firstFrameMouse = true;

    _device       = view.device; // get access to the GPU
    _commandQueue = [_device newCommandQueue]; // "conveyer belt" for command buffers per frame

    // load the compiled shaders into the library of shaders
    id<MTLLibrary> library = [_device newDefaultLibrary];
    if (library == nil) {
    	NSLog(@"Failed to find default library.");
        return nil;
    }

    _frameDataBuffer = [_device newBufferWithLength:sizeof(FrameData)
                                options:MTLResourceStorageModeShared];
    _sim = [[FluidSim alloc] initWithDevice:_device
                                     library:library
                                commandQueue:_commandQueue
                                       width:simWidth
                                      height:simHeight];

    // ---- set up rendering pipelines ----
    id<MTLFunction> vertexFn = [library newFunctionWithName:@"vertex_main"];
    MTLRenderPipelineDescriptor *pipeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipeDesc.vertexFunction = vertexFn;
    pipeDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat;

    NSError *error = nil;

    // Render Pipeline (Smoke)
    pipeDesc.fragmentFunction = [library newFunctionWithName:@"fragment_smoke"];
    _renderPipeline = [_device newRenderPipelineStateWithDescriptor:pipeDesc error:&error];

    _fpsLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(10, view.frame.size.height - 30, 200, 20)];
    [_fpsLabel setEditable:NO];
    [_fpsLabel setSelectable:NO];
    [_fpsLabel setBezeled:NO];
    [_fpsLabel setDrawsBackground:NO];
    [_fpsLabel setTextColor:[NSColor whiteColor]];
    [_fpsLabel setFont:[NSFont monospacedDigitSystemFontOfSize:14 weight:NSFontWeightRegular]];
    [view addSubview:_fpsLabel];

    return self;
}

- (void)updateMousePos:(NSEvent *)event {
    NSPoint p = [event locationInWindow];
    _mousePos = simd_make_float2(p.x / _width, p.y / _height);
}

- (void)mouseMoved:(NSEvent *)event {
    [self updateMousePos:event];
}
- (void)mouseDragged:(NSEvent *)event {
    [self updateMousePos:event];
}
- (void)mouseDown:(NSEvent *)event {
    _leftDown = true;
    [self updateMousePos:event];
}
- (void)mouseUp:(NSEvent *)event {
    _leftDown = false;
    [self updateMousePos:event];
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = event.scrollingDeltaY;
    SimConstants *c = (SimConstants *)_sim.simConstantsBuffer.contents;
    c->mouseRadius += delta * 0.001f;
    if (c->mouseRadius < 0.005f) c->mouseRadius = 0.005f;
    if (c->mouseRadius > 0.5f) c->mouseRadius = 0.5f;
}

- (void)keyDown:(NSEvent *)event {
    [self handleKeyEvent:event isDown:YES];
}

- (void)keyUp:(NSEvent *)event {
    [self handleKeyEvent:event isDown:NO];
}

- (void)handleKeyEvent:(NSEvent *)event isDown:(BOOL)isDown {
    if (!isDown) return;

    NSString *chars = [event charactersIgnoringModifiers];
    if ([chars isEqualToString:@"b"]) {
        _isSolidMode = !_isSolidMode;
    } else if ([chars isEqualToString:@"r"]) {
        [_sim reset:_commandQueue];
    } else if ([chars isEqualToString:@" "]) {
        _paused = !_paused;
    }
}

- (void)handleDroppedImage:(NSImage *)image {
    [_sim setSmokeImage:image];
}

// drawInMTKView method -- render loop for the program, called every frame
- (void)drawInMTKView:(MTKView *)view {
    NSTimeInterval currentTime = CACurrentMediaTime();
    double dt = currentTime - _lastFrameTime;
    _lastFrameTime = currentTime;

    if (dt > 0 && dt < 1.0) {
        double instantFPS = 1.0 / dt;
        // Exponential smoothing: 95% old value, 5% new value
        _smoothedFPS = (_smoothedFPS * 0.95) + (instantFPS * 0.05);
        _fpsLabel.stringValue = [NSString stringWithFormat:@"FPS: %.0f", _smoothedFPS];
    }

    if (_firstFrameMouse && _leftDown) {

        _lastMousePos = _mousePos;
        _firstFrameMouse = false;
    }

    // update per-frame data on CPU side
    FrameData *frameData = (FrameData *)_frameDataBuffer.contents;
    frameData->mouse.pos = _mousePos;
    frameData->mouse.delta = _mousePos - _lastMousePos;
    frameData->mouse.leftDown = _leftDown;
    frameData->mouse.isSolidMode = _isSolidMode;
    frameData->mouse.isPaused = _paused;

    _lastMousePos = _mousePos;

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer]; // list of commands to the gpu

    // ---- compute pass, all sim steps ----
    id<MTLComputeCommandEncoder> comp = [commandBuffer computeCommandEncoder];
    [_sim encodeSimStep:comp frameData:_frameDataBuffer];
    [comp endEncoding];

    // ---- render pass ----
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor; // what pass i'm rendering to
    if (renderPassDescriptor != nil) {
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        [renderEncoder setRenderPipelineState:_renderPipeline];

        [renderEncoder setFragmentTexture:[_sim smokeTexture] atIndex:0];
        [renderEncoder setFragmentTexture:[_sim solidsTexture] atIndex:1];
        [renderEncoder setFragmentBuffer:_sim.simConstantsBuffer offset:0 atIndex:0];
        [renderEncoder setFragmentBuffer:_frameDataBuffer offset:0 atIndex:1];

        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        [renderEncoder endEncoding];
        // tell metal to display this frame's texture on screen when GPU finishes
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    [commandBuffer commit]; // send command buffer to metal
    // GPU is now working async, CPU has done its job
}

// called when the window size changes. empty for now
- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size { // (TODO): fill this in
}

@end
