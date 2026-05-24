#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface FluidSim : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                         width:(int)width
                        height:(int)height;

- (void)encodeSimStep:(id<MTLComputeCommandEncoder>)encoder frameData:(id<MTLBuffer>)frameData;
- (void)reset:(id<MTLCommandQueue>)commandQueue;
- (void)setSmokeImage:(NSImage *)image;
- (id<MTLTexture>)smokeTexture;
- (id<MTLTexture>)solidsTexture;
- (id<MTLBuffer>)simConstantsBuffer; // getter for Renderer
@end

#endif
