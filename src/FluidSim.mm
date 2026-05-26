#import "../include/FluidSim.hpp"
#import "../include/SimParams.h"
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@implementation FluidSim {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;

    // sim textures
    id<MTLTexture> _velX;
    id<MTLTexture> _velXTemp;
    id<MTLTexture> _velY;
    id<MTLTexture> _velYTemp;
    id<MTLTexture> _pressure;
    id<MTLTexture> _smoke; // smokeMap analog from CPU version
    id<MTLTexture> _smokeTemp;
    id<MTLTexture> _smokeInitial;
    id<MTLTexture> _solids;
    id<MTLTexture> _pressureSolveData;

    // compute pipelines (one needed for each kernel)
    id<MTLComputePipelineState> _injectSmokePipeline; // pipeline for drawing smoke
    id<MTLComputePipelineState> _injectVelocityPipeline; // pipeline for changing velocity
    id<MTLComputePipelineState> _injectSolidsPipeline;
    id<MTLComputePipelineState> _advectVelXPipeline;
    id<MTLComputePipelineState> _advectVelYPipeline;
    id<MTLComputePipelineState> _advectSmokePipeline;
    id<MTLComputePipelineState> _clearTexturesPipeline;
    id<MTLComputePipelineState> _initSolidsPipeline;
    id<MTLComputePipelineState> _updateVelocitiesPipeline;
    id<MTLComputePipelineState> _precomputePressureDataPipeline;
    id<MTLComputePipelineState> _scaleTexturePipeline;
    id<MTLComputePipelineState> _copyTexturePipeline;
    // red black Gauss-Seidel
    id<MTLComputePipelineState> _gsRedPipeline;
    id<MTLComputePipelineState> _gsBlackPipeline;

    id<MTLBuffer> _simConstantsBuffer;
    int _width;
    int _height;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  commandQueue:(id<MTLCommandQueue>)commandQueue
                         width:(int)width
                        height:(int)height {
    self = [super init];
    if (!self) return nil;

    _device = device;
    _commandQueue = commandQueue;
    _width  = width;
    _height = height;

    [self allocateTextures];
    [self buildPipelines:library];
    [self buildConstantsBuffer];
    [self clearTextures:commandQueue];
    [self initializeSolids:commandQueue];

    return self;
}

- (id<MTLBuffer>)simConstantsBuffer {
    return _simConstantsBuffer;
}

- (id<MTLTexture>)makeTextureWidth:(int)w height:(int)h {
    MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = MTLPixelFormatR32Float;
    td.width       = w;
    td.height      = h;
    td.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:td];
}

- (void)allocateTextures {
    _velX        = [self makeTextureWidth:_width + 1 height:_height];
    _velXTemp    = [self makeTextureWidth:_width + 1 height:_height];
    _velY        = [self makeTextureWidth:_width     height:_height + 1];
    _velYTemp    = [self makeTextureWidth:_width     height:_height + 1];
    _pressure    = [self makeTextureWidth:_width     height:_height];

    // smoke textures use RGBA32Float
    MTLTextureDescriptor *smokeDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                           width:_width
                                                          height:_height
                                                       mipmapped:NO];
    smokeDesc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    smokeDesc.storageMode = MTLStorageModePrivate;
    _smoke     = [_device newTextureWithDescriptor:smokeDesc];
    _smokeTemp = [_device newTextureWithDescriptor:smokeDesc];
    _smokeInitial = [_device newTextureWithDescriptor:smokeDesc];

    // solids uses R8Uint
    MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = MTLPixelFormatR8Uint;
    td.width       = _width;
    td.height      = _height;
    td.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    _solids = [_device newTextureWithDescriptor:td];

    // pressureSolveData uses RG32Float
    MTLTextureDescriptor *psdDesc = [[MTLTextureDescriptor alloc] init];
    psdDesc.textureType = MTLTextureType2D;
    psdDesc.pixelFormat = MTLPixelFormatRG32Float;
    psdDesc.width       = _width;
    psdDesc.height      = _height;
    psdDesc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    psdDesc.storageMode = MTLStorageModePrivate;
    _pressureSolveData = [_device newTextureWithDescriptor:psdDesc];
}

- (id<MTLComputePipelineState>)makePipeline:(id<MTLLibrary>)library name:(NSString *)name {
    NSError *error = nil;
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) { NSLog(@"Kernel not found: %@", name); return nil; }
    id<MTLComputePipelineState> pipeline = [_device newComputePipelineStateWithFunction:fn error:&error];
    if (error) NSLog(@"Pipeline error for %@: %@", name, error);
    return pipeline;
}

- (void)buildPipelines:(id<MTLLibrary>)library {
    _injectVelocityPipeline   = [self makePipeline:library name:@"inject_velocity"];
    _injectSmokePipeline      = [self makePipeline:library name:@"inject_smoke"];
    _injectSolidsPipeline     = [self makePipeline:library name:@"inject_solids"];
    _advectVelXPipeline       = [self makePipeline:library name:@"advect_velX"];
    _advectVelYPipeline       = [self makePipeline:library name:@"advect_velY"];
    _advectSmokePipeline      = [self makePipeline:library name:@"advect_smoke"];
    _gsRedPipeline            = [self makePipeline:library name:@"gs_red"];
    _gsBlackPipeline          = [self makePipeline:library name:@"gs_black"];
    _updateVelocitiesPipeline = [self makePipeline:library name:@"update_velocities"];
    _precomputePressureDataPipeline = [self makePipeline:library name:@"precompute_pressure_data"];
    _scaleTexturePipeline     = [self makePipeline:library name:@"scale_texture"];
    _copyTexturePipeline      = [self makePipeline:library name:@"simple_copy"];
    _clearTexturesPipeline    = [self makePipeline:library name:@"clear_textures"];
    _initSolidsPipeline       = [self makePipeline:library name:@"init_solids"];
}

- (void)buildConstantsBuffer {
    SimConstants c;
    c.cellSize     = 1.f / fmin(_width, _height);
    c.deltaTime    = 1.f / 30.f;
    c.fluidDensity = 0.3f;
    c.width        = _width;
    c.height       = _height;
    c.mouseRadius = 0.03f;
    c.velocityStrength = 1.8f;
    c.velDamping = 0.98;
    c.weightSOR = 1.7f;
    _simConstantsBuffer = [_device newBufferWithBytes:&c
                                               length:sizeof(SimConstants)
                                              options:MTLResourceStorageModeShared];
}

// helper for flexible kernel dispatch
- (void)dispatch:(id<MTLComputeCommandEncoder>)encoder
        pipeline:(id<MTLComputePipelineState>)pipeline
            grid:(MTLSize)grid
        textures:(NSArray<id<MTLTexture>> *)textures
         buffers:(NSArray<id<MTLBuffer>> *)buffers {
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < textures.count; i++) {
        [encoder setTexture:textures[i] atIndex:i];
    }
    for (NSUInteger i = 0; i < buffers.count; i++) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }

    MTLSize threadgroup = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
}

// Helper to perform a compute pass and wait for completion
- (void)computePass:(void (^)(id<MTLComputeCommandEncoder> encoder))block {
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    block(encoder);
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

- (void)swapTextures:(id<MTLTexture> *)a with:(id<MTLTexture> *)b {
    id<MTLTexture> temp = *a;
    *a = *b;
    *b = temp;
}

- (void)encodeSimStep:(id<MTLComputeCommandEncoder>)encoder
            frameData:(id<MTLBuffer>)frameData {
    MTLSize grid = MTLSizeMake(_width, _height, 1);

    // inject (mouse input) - always run so user can draw even when paused
    [self dispatch:encoder
          pipeline:_injectVelocityPipeline
              grid:MTLSizeMake(_width+1, _height+1, 1)
          textures:@[_velX, _velY]
           buffers:@[_simConstantsBuffer, frameData]];
    [self dispatch:encoder
          pipeline:_injectSmokePipeline
              grid:grid
          textures:@[_smoke]
           buffers:@[_simConstantsBuffer, frameData]];
    [self dispatch:encoder
          pipeline:_injectSolidsPipeline
              grid:grid
          textures:@[_solids]
           buffers:@[_simConstantsBuffer, frameData]];

    // Check if paused from frameData
    FrameData *fd = (FrameData *)frameData.contents;
    if (fd->mouse.isPaused) return;

    // advect velocities
    [self dispatch:encoder
          pipeline:_advectVelXPipeline
              grid:MTLSizeMake(_width+1, _height, 1)
          textures:@[_velX, _velY, _solids, _velXTemp]
           buffers:@[_simConstantsBuffer]];
    [self swapTextures:&_velX with:&_velXTemp];

    [self dispatch:encoder
          pipeline:_advectVelYPipeline
              grid:MTLSizeMake(_width, _height+1, 1)
          textures:@[_velX, _velY, _solids, _velYTemp]
           buffers:@[_simConstantsBuffer]];
    [self swapTextures:&_velY with:&_velYTemp];

    // advect smoke
    [self dispatch:encoder
          pipeline:_advectSmokePipeline
              grid:grid
          textures:@[_velX, _velY, _smoke, _solids, _smokeTemp]
           buffers:@[_simConstantsBuffer]];
    [self swapTextures:&_smoke with:&_smokeTemp];

    // precompute pressure data
    [self dispatch:encoder
          pipeline:_precomputePressureDataPipeline
              grid:grid
          textures:@[_velX, _velY, _solids, _pressureSolveData]
           buffers:@[_simConstantsBuffer]];

    // pressure solve
    for (int i = 0; i < 80; i++) {
        [self dispatch:encoder
              pipeline:_gsRedPipeline
                  grid:grid
              textures:@[_pressure, _pressureSolveData]
               buffers:@[_simConstantsBuffer]];
        [self dispatch:encoder
              pipeline:_gsBlackPipeline
                  grid:grid
              textures:@[_pressure, _pressureSolveData]
               buffers:@[_simConstantsBuffer]];
    }

    // update velocities (dispatch enough threads to cover width+1 / height+1)
    [self dispatch:encoder
          pipeline:_updateVelocitiesPipeline
              grid:MTLSizeMake(_width + 1, _height + 1, 1)
          textures:@[ _velX, _velY, _pressure, _solids ]
           buffers:@[ _simConstantsBuffer ]];
}

- (void)reset:(id<MTLCommandQueue>)commandQueue {
    [self computePass:^(id<MTLComputeCommandEncoder> enc) {
        // Clear everything
        [self dispatch:enc
              pipeline:_clearTexturesPipeline
                  grid:MTLSizeMake(_width + 1, _height + 1, 1)
              textures:@[_velX, _velXTemp, _velY, _velYTemp, _pressure, _smoke, _smokeTemp, _solids, _pressureSolveData]
               buffers:@[_simConstantsBuffer]];

        // Restore smoke from _smokeInitial using simple_copy (no flip)
        [enc setComputePipelineState:_copyTexturePipeline];
        MTLSize gridSize = MTLSizeMake(_smoke.width, _smoke.height, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

        [enc setTexture:_smokeInitial atIndex:0];
        [enc setTexture:_smoke atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        [enc setTexture:_smokeTemp atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }];

    [self initializeSolids:commandQueue];
}

- (void)initializeSolids:(id<MTLCommandQueue>)commandQueue {
    [self computePass:^(id<MTLComputeCommandEncoder> enc) {
        [self dispatch:enc
              pipeline:_initSolidsPipeline
                  grid:MTLSizeMake(_width, _height, 1)
              textures:@[_solids]
               buffers:@[_simConstantsBuffer]];
    }];
}

- (void)clearTextures:(id<MTLCommandQueue>)commandQueue {
    [self computePass:^(id<MTLComputeCommandEncoder> enc) {
        [self dispatch:enc
              pipeline:_clearTexturesPipeline
                  grid:MTLSizeMake(_width + 1, _height + 1, 1)
              textures:@[_velX, _velXTemp, _velY, _velYTemp, _pressure, _smoke, _smokeTemp, _solids, _pressureSolveData]
               buffers:@[_simConstantsBuffer]];
    }];
}

- (void)setSmokeImage:(NSImage *)image {
    if (!image) return;

    NSError *error = nil;
    MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice:_device];

    NSDictionary *options = @{
        MTKTextureLoaderOptionSRGB: @NO,
        MTKTextureLoaderOptionTextureUsage: @(MTLTextureUsageShaderRead)
    };

    CGImageRef cgImage = [image CGImageForProposedRect:NULL context:NULL hints:NULL];
    id<MTLTexture> tempTexture = [loader newTextureWithCGImage:cgImage options:options error:&error];

    if (error || !tempTexture) {
        NSLog(@"Failed to load image texture: %@", error);
        return;
    }

    [self computePass:^(id<MTLComputeCommandEncoder> encoder) {
        [encoder setComputePipelineState:_scaleTexturePipeline];

        MTLSize gridSize = MTLSizeMake(_smoke.width, _smoke.height, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

        // Scale into _smoke
        [encoder setTexture:tempTexture atIndex:0];
        [encoder setTexture:_smoke atIndex:1];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Scale into _smokeTemp
        [encoder setTexture:_smokeTemp atIndex:1];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Scale into _smokeInitial
        [encoder setTexture:_smokeInitial atIndex:1];
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }];
}

- (id<MTLTexture>)smokeTexture {
    return _smoke;
}

- (id<MTLTexture>)solidsTexture {
    return _solids;
}

@end
